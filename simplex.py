"""
Aluno: Enok Januário da Rocha
Professor: Cristiano Arbex
Disciplina: Pesquisa Operacional
UNIVERSIDADE FEDERAL DE MINAS GERAIS
"""


import sys
import numpy as numpy
import fractions as frac

numpy.set_printoptions(formatter={'all':lambda x: str(frac.Fraction(x).limit_denominator())})

'''
Dizemos que uma PL encontra-se na Forma Padrão de Igualdade (FPI) quando é um problema de maximização e temos 
somente restrições de igualdade. Nem sempre recebemos uma PL nesse formato, então é nosso dever adequá-la antes
de aplicar o simplex.
'''

class FormaPadrao(object):

	def __init__(self, A, b, c, sinais, RestricoesNegatividade):

		'''
		folga se refere às variáveis de folga que são acrescentadas à PL a fim de transformar inequações em equações.
		Quando recebemos uma PL que possui restrição do tipo <= acrescentamos uma variável de folga e substituímos por =
		Quando recebemos uma PL que possui restrição do tipo >= acrescentamos uma variável de excesso e substituímos por =
		Quando recemos uma PL que possui restrição do tipo == não fazemos nada
		'''

		self.folga = []
		self.vars = {i: str(i) for i in range(A.shape[1])}

		'''

		A é o "recheio do tableau" que será preenchido com os coeficientes das variáveis das restrições
		c é o vetor de coeficientes da função objetivo que queremos maximizar
		b é o vetor de coeficientes livres ao lado diretiro dos sinais de igualdade
		
		'''

		self.A = A
		self.c = c 
		self.b = b

		'''
		As restrições de Negatividade nos dizem se uma variável pode assumir um valor negativo ou não. Em FPI todas as variáveis
		obrigatoriamente devem possuir esse tipo de restrição. Quando recebemos uma PL em que uma ou mais variáveis não possuem 
		devemos tratar e modificar o tableau
		'''

		self.VerificaDominio(RestricoesNegatividade)
		self._VariaveisDeFolga(sinais)

	def VerificaDominio(self, RestricoesNegatividade):

		for i, sinal in enumerate(RestricoesNegatividade):
			if sinal == '0':
				self.c = numpy.append(self.c, -1 * self.c[i])
				self.A = numpy.append(self.A, numpy.array([-1 * self.A[:, i]]).T, 1)
				self.vars[i] = self.vars[i] + '-' + str(self.A.shape[1] - 1)
	
	#A seguinte função foi criada para verificar a dependencia de restrições
	
	def DependenciaDeRestricoes(self):
		nVariaveis, nRestricoes = self.A.shape
		if nVariaveis <= nRestricoes and nVariaveis != numpy.linalg.matriz_rank(self.A):
			return True
		return False

	'''
	Como já mencionado no início do código, as variáveis de folga/excesso desempenham importante função na transformação de uma PL
	para o formato FPI. Elas representam a diferença existente entre o sinal de inequação referente e a igualdade, de certa forma.
	Quando acrescentadas ao tableau, possuem coeficiente 1 ou -1 dependendo se é de folga ou excesso.
	'''

	def _VariaveisDeFolga(self, sinais):
		for i, sinal in enumerate(sinais):
			nLinhas = self.A.shape[0]
			if sinal == '<=':
				'''
				Nesta parte do algoritmo verificamos se trata-se e um sinal de <=. Neste caso, devemos inserir uma variável de folga que 
				possui coeficiente unitário positivo no tableau e, coo mencionado anteriormente, não aparece em nenhuma outra restrição e 
				nem no vetor c.
				'''
				self.A = numpy.append(self.A, 1 * numpy.zeros(shape = [nLinhas, 1]), 1)
				self.A[i, -1] = 1
				self.c = numpy.append(self.c, 0)
				self.folga.append(self.A.shape[1] - 1)

			elif sinal == '>=':

				'''
				No caso do >= necessitamos de uma variável de excesso. Variáveis de excesso possuem coenficiente igual a -1. Como 
				a mesma aparece somente nesta restrição, ela possuirá coeficiente nulo para as outras. Vale salientar tambem que 
				no vetor c, todas as variáveis de folga e excesso são descartáveis, isto é, possuem peso nulo.
				'''
				self.A = numpy.append(self.A, -1 * numpy.zeros(shape = [nLinhas, 1]), 1)
				self.A[i, -1] = -1
				self.c = numpy.append(self.c, 0)
				self.folga.append(self.A.shape[1] - 1)

	def Tableau(self):
		"""
		Glossário de variáveis:
		artificiais = vetor que armazena as variáveis artificiais (variáveis acrescentadas em caso de variável sem restrição de não negatividade)
		SBasicaV = recebe as variáveis que formam a base 
		"""
		artificiais = []
		SBasicaV = []
		Bnovo = numpy.array(self.b)
		nLinhas, nColunas = self.A.shape
		nArtificiais = min(nLinhas, nColunas)
		tableau = numpy.array(self.A)

		tableau = tableau + frac.Fraction()

		for linha, coluna in enumerate(self.folga):
			if tableau[linha, coluna] == -1 and Bnovo[linha] < 0:
				nArtificiais -= 1
				tableau[linha] = -1 * tableau[linha]
				Bnovo[linha] = -1 * Bnovo[linha]
				SBasicaV.append((linha, coluna))
			
			elif tableau[linha, coluna] == 1 and Bnovo[linha] > 0:
				nArtificiais -= 1
				SBasicaV.append((linha, coluna))

		tableau = numpy.append(tableau, numpy.zeros(shape=[nLinhas, nArtificiais]), 1)

		SBasicaV_nLinhas = set(map(lambda x: x[0], SBasicaV))
		nLinhas, nColunas = tableau.shape
		valArtificial = 0
		
		for i in range(0, nLinhas):
			if i in SBasicaV_nLinhas:
				continue
			artificiais.append((i, nColunas - nArtificiais + valArtificial))
			SBasicaV.append((i, nColunas - nArtificiais + valArtificial))
			if tableau[i, -1] < 0:
				tableau[i, :] = -1 * tableau[i, :]
			tableau[i, nColunas -nArtificiais + valArtificial] = 1
			valArtificial +=1

		return numpy.column_stack((tableau, Bnovo)), artificiais, SBasicaV

	"""
	O tipo Fraction foi usado com a finalidade de diminuir erro de truncamento durantes as 
	fases de execução do algoritmo.
	"""

	"""
	def converteVariaveis(self, x):
		assert len(x) == self.A.shape[1]
		ans = numpy.zeros(len(self.vars))
		for i in range(len(self.vars)):
			varsPadrao = self.vars[i].split('-')
			if len(varsPadrao) > 1:
				ans[i] -= x[int(varsPadrao[1])]
			if varsPadrao[0] != '':
				ans[i] += x[int(varsPadrao[0])]
		assert len(ans) == len(self.vars)
		return frac.Fraction(ans)
	"""



"""
Implementação do algoritmo propriamente dito que irá rer nossas PL's
"""
class Simplex:
		
	def Resolve(self, PL):
		tableau, varsArtificiais, SBasicaV = PL.Tableau()
		n, m = tableau.shape

		if len(varsArtificiais) != 0:
			viavel, tableauVals  = self.Fase1(tableau, varsArtificiais, SBasicaV)
			tableau, SBasicaV = tableauVals
			if not viavel:
				return 'Inviavel', None, None

			tableau = numpy.column_stack((tableau[:, 0:m-len(varsArtificiais) - 1], tableau[:, -1]))

		tableau, obj, SBasicaV, limitada = self.Fase2(tableau, PL.c, SBasicaV)
		if limitada:
			nVariaveis = tableau.shape[1] - 1
			x = numpy.zeros(nVariaveis)
			for i, coord in enumerate(SBasicaV):
				linha, coluna = coord
				x[coluna] = tableau[linha, -1]
			return 'Resolvido', obj[-1], x
		else:
			return 'Ilimitada', None, None


	def Fase1(self, tableau, varsArtificiais, SBasicaV):
		#print(n,m)
		n, m = tableau.shape

		c = numpy.zeros(m - 1)
		nColunasArtificiais = list(map(lambda x: x[1], varsArtificiais))
		c[nColunasArtificiais] = -1 # deve ser -1
		obj = self.CalculaVO(tableau, c, SBasicaV)
		tableau, obj, SBasicaV, limitada = self._simplex(tableau, obj, SBasicaV)

		if numpy.isclose(obj[-1], 0):
			return True, (tableau, SBasicaV)
		else:
			return False, (None, None)

	def Fase2(self, tableau, c, SBasicaV):
		#print(n,m)
		n, m = tableau.shape
		obj = self.CalculaVO(tableau, c, SBasicaV)
		tableau, obj, SBasicaV, limitada = self._simplex(tableau, obj, SBasicaV)
		return tableau, obj, SBasicaV, limitada

	def _simplex(self, tableau, obj, SBasicaV):

		while True:

			negativos = numpy.where(obj[:-1] < 0)[0]
			if len(negativos) == 0:
				break

			novaBase = negativos[0]

			linha = -1
			custominimo = float('Inf')
			for i in range(tableau.shape[0]):
				if tableau[i, novaBase] > 0:
					custo = (tableau[i, -1]) / (tableau[i, novaBase])
					if custo < custominimo:
						custominimo = custo
						linha = i

			if linha == -1:
				return tableau, obj, SBasicaV, False

			saida = list(filter(lambda x: x[0] == linha, SBasicaV))
			tableau, obj = self.Pivotea(tableau, obj, linha, novaBase)
			assert len(saida) == 1
			SBasicaV.remove(saida[0])
			SBasicaV.append((linha, novaBase))

		return tableau, obj, SBasicaV, True

	def CalculaVO(self, tableau, c, SBasicaV):
		
		n, m = tableau.shape

		obj = numpy.append(c, 0)
		for coord in SBasicaV:
			linha, coluna = coord
			obj = obj - obj[coluna] * tableau[linha, :]
		obj = -1 * obj 

		return obj

	def Pivotea(self, tableau, obj, linha, coluna):

		tableau[linha, :] = (tableau[linha, :]) / (tableau[linha, coluna])
		nLinhas, nColunas = tableau.shape

		for r in range(0, nLinhas):
			if r != linha:
				tableau[r, :] = tableau[r, :] - tableau[r, coluna]*tableau[linha, :]
				obj = obj - obj[coluna] * tableau[linha, :]

		return tableau, obj

"""
def LEPL(filepath):
file = open(filepath, 'r')
	nVariaveis = int(file.readline()) 
	nRestricoes = int(file.readline()) 

	negRestricoes = numpy.array(file.readline().split(' ')).astype("int")

	c = file.readline().split(' ')
	if c[-1] == '\n':
		c = c[:-1]
	
	c = [frac.Fraction(float(c[i])) for i in range(len(c))]
	matriz = []
	for i in range(nRestricoes):
		linha = file.readline().split(' ')
		if linha[-1][-1] == '\n':
			linha[-1] = linha[-1][:-1] # Remove \n
		matriz.append(linha)
	file.close()
"""

def LePL(filepath):

	file = open(filepath, 'r')
	nVariaveis = int(file.readline()) 
	nRestricoes = int(file.readline()) 

	negRestricoes = numpy.array(file.readline().split(' ')).astype("int")

	c = file.readline().split(' ')
	if c[-1] == '\n':
		c = c[:-1]
	
	c = [frac.Fraction(float(c[i])) for i in range(len(c))]
	matriz = []
	for i in range(nRestricoes):
		linha = file.readline().split(' ')
		if linha[-1][-1] == '\n':
			linha[-1] = linha[-1][:-1] # Remove \n
		matriz.append(linha)
	file.close()

	b = [matriz[i][-1] for i in range(len(matriz))]
	sinais = [matriz[i][-2] for i in range(len(matriz))]
	matriz = [matriz[i][:-2] for i in range(len(matriz))]

	b = numpy.array([frac.Fraction(float(b[i])) for i in range(len(b))])

	for i in range(len(matriz)):
		for j in range(len(matriz[i])):
			matriz[i][j] = frac.Fraction(float(matriz[i][j]))

	A = numpy.zeros(shape=(nRestricoes, nVariaveis));

	for i in range(len(matriz)):
		for j in range(len(matriz[i])):
			A[i][j] = frac.Fraction(float(matriz[i][j]));
		
	file.close()
	
	return nVariaveis, nRestricoes, A, b, c, sinais, negRestricoes

#VERIFICAR
def escrevePL(filepath, solucao, obj, x):
	file = open(filepath, 'w')

	if solucao == "Resolvido":
		file.write("Status: otimo\n")
		file.write("Objetivo: " + format(obj, '.2f') + "\n")
		file.write("Solucao: \n")
		for item in x:
			file.write(str(format(item, '.2f'))+'\t')
		file.write("\nCertificado: \n")
		file.write("\n")
	elif solucao == "Ilimitada":
		file.write("Status: ilimitado\n")
		file.write("Certificado: \n")
		file.write("\n")
	elif solucao == "Inviavel":
		file.write("Status: inviavel\n")
		file.write("Certificado: \n")
		file.write("\n")
	file.close()


if __name__ == '__main__':
	
	"""
	
	Verifica se o arquivo atende as especificações e se os parâmetros estão corretos. Caso haja algum problema com o arquivo, o sistema irá exibir uma mensagem de erro e crashar.
	
	"""

	if len(sys.argv) < 3:
		print("O número de argumentos é inválido. Favor verificar o nome do arquivo de entrada e o de saída.")
		sys.exit()

	nVariaveis, nRestricoes, A, b, c, sinais, negRestricoes = LePL(sys.argv[1])

	PL = FormaPadrao(A, b, c, sinais, negRestricoes)
	r = Simplex()
	solucao, obj, x = r.Resolve(PL)

	"""
	Escreve os resultados encontrados no arquivo passado como parâmetro
	"""
	escrevePL(sys.argv[2], solucao, obj, x)
