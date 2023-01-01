import os


def menu():
    os.system('cls')
    print(
        '\n' + '▒'*122,
        '\n' + '▒'*122,
        '\n' + '▒'*34 + ' ╔' + '═'*50 + '╗ ' + '▒'*34,
        '\n' + '▒'*34 + ' ║' + '  1. CALCULAR RUTA ENTRE ORIGEN Y DESTINO' + ' '*9 + '║ ' + '▒'*34,
        '\n' + '▒'*34 + ' ╠' + '═'*50 + '╣ ' + '▒'*34,
        '\n' + '▒'*34 + ' ║' + '  2. VISUALIZAR EL MAPA DE MADRID' + ' '*17 + '║ ' + '▒'*34,
        '\n' + '▒'*34 + ' ╠' + '═'*50 + '╣ ' + '▒'*34,
        '\n' + '▒'*34 + ' ║' + '  3. PINTAR UNA CALLE ESPECÍFICA' + ' '*18 + '║ ' + '▒'*34,
        '\n' + '▒'*34 + ' ╠' + '═'*50 + '╣ ' + '▒'*34,
        '\n' + '▒'*34 + ' ║' + '  4. HELP (SOLO SI DA TIEMPO)' + ' '*21 + '║ ' + '▒'*34,
        '\n' + '▒'*34 + ' ╠' + '═'*50 + '╣ ' + '▒'*34,
        '\n' + '▒'*34 + ' ║' + '  5. EXIT' + ' '*41 + '║ ' + '▒'*34,
        '\n' + '▒'*34 + ' ╚' + '═'*50 + '╝ ' + '▒'*34,
        '\n' + '▒'*122,
        '\n' + '▒'*122,
    )
    x = input('\n¿Qué desea hacer? (1-5): ')
    while x not in ['1', '2', '3', '4', '5']:
        print('\nPor favor, elija una opción correcta')
        x = input('\n¿Qué desea hacer? (1-5): ')
    return int(x)


if __name__ == '__main__':
    menu()