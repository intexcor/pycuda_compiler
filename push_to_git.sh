#!/bin/bash
# Скрипт для push в приватный Git репозиторий

set -e

echo "============================================================"
echo "Настройка Git репозитория для push"
echo "============================================================"

# Проверяем, есть ли уже remote
if git remote | grep -q origin; then
    echo "✓ Remote 'origin' уже настроен"
    git remote -v
else
    echo ""
    echo "У вас нет настроенного remote репозитория."
    echo ""
    echo "Чтобы создать приватный репозиторий:"
    echo ""
    echo "1. GitHub:"
    echo "   - Перейдите на https://github.com/new"
    echo "   - Создайте новый репозиторий (выберите Private)"
    echo "   - НЕ инициализируйте его (не добавляйте README, .gitignore и т.д.)"
    echo "   - Скопируйте URL репозитория (например: https://github.com/username/pycuda_compiler.git)"
    echo ""
    echo "2. GitLab:"
    echo "   - Перейдите на https://gitlab.com/projects/new"
    echo "   - Создайте новый проект (выберите Private)"
    echo "   - Скопируйте URL репозитория"
    echo ""
    read -p "Введите URL вашего приватного репозитория: " REPO_URL
    
    if [ -z "$REPO_URL" ]; then
        echo "Ошибка: URL не может быть пустым"
        exit 1
    fi
    
    echo ""
    echo "Добавляю remote..."
    git remote add origin "$REPO_URL"
    echo "✓ Remote добавлен: $REPO_URL"
fi

echo ""
echo "Текущая ветка: $(git branch --show-current)"
echo ""

# Переименовываем в main если нужно
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" = "master" ]; then
    read -p "Переименовать ветку master в main? (y/n): " RENAME
    if [ "$RENAME" = "y" ] || [ "$RENAME" = "Y" ]; then
        git branch -m master main
        echo "✓ Ветка переименована в main"
    fi
fi

echo ""
echo "Проверяю статус..."
git status

echo ""
read -p "Выполнить push? (y/n): " PUSH
if [ "$PUSH" = "y" ] || [ "$PUSH" = "Y" ]; then
    BRANCH=$(git branch --show-current)
    echo ""
    echo "Выполняю push в origin/$BRANCH..."
    echo "Примечание: Вам может потребоваться ввести credentials"
    git push -u origin "$BRANCH"
    echo ""
    echo "✓ Push выполнен успешно!"
else
    echo ""
    echo "Для ручного push выполните:"
    echo "  git push -u origin $(git branch --show-current)"
fi
