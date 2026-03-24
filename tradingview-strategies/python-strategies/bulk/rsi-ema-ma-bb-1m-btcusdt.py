#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "Rsi Ema Ma Bollinger Bands 1m Btcusdt"
timeframe = "1m"
leverage = 1

def calculate_sma(series, length):
    return series.rolling(window=length).mean()

def calculate_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def calculate_rma(series, length):
    # Wilder's Smoothing (RMA) used in Pine RSI
    # alpha = 1 / length
    return series.ewm(alpha=1.0/length, adjust=False).mean()

def calculate_rsi(close, length):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = calculate_rma(gain, length)
    avg_loss = calculate_rma(loss, length)
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_std(series, length):
    return series.rolling(window=length).std(ddof=1)

def generate_signals(prices):
    # Ensure DataFrame format for column access
    if not isinstance(prices, pd.DataFrame):
        prices = pd.DataFrame(prices)
    
    close = prices['close']
    open_price = prices['open']
    
    n = len(close)
    signals = np.zeros(n, dtype=int)
    
    # Minimum period check to avoid NaN errors
    if n < 75:
        return signals
    
    # Bollinger Bands
    bb_len = 5
    bb_mult = 1.5
    basis = calculate_sma(close, bb_len)
    std_dev = calculate_std(close, bb_len)
    upper = basis + (bb_mult * std_dev)
    lower = basis - (bb_mult * std_dev)
    
    # RSI
    rsi_len = 7
    rsi_low_len = 25
    rsi_high_len = 75
    myrsi = calculate_rsi(close, rsi_len)
    myrsi2 = calculate_rsi(close, rsi_low_len)
    myrsi3 = calculate_rsi(close, rsi_high_len)
    
    # SMA
    sma_high_len = 10
    sma_low_len = 5
    myma = calculate_sma(close, sma_high_len)
    myma2 = calculate_sma(close, sma_low_len)
    
    # EMA
    ema_high_len = 30
    ema_low_len = 20
    myema = calculate_ema(close, ema_high_len)
    myema2 = calculate_ema(close, ema_low_len)
    
    # Helper functions for crossover/crossunder
    def crossover(a, b):
        return (a.shift(1) <= b.shift(1)) & (a > b)
    
    def crossunder(a, b):
        return (a.shift(1) >= b.shift(1)) & (a < b)
    
    # Long Entry Conditions (idunno)
    cond_long = (
        (close < myma2) &
        (close < myma) &
        (close < myema) &
        (close < myema2) &
        crossunder(close, lower) &
        crossunder(myrsi, myrsi2) &
        crossunder(close, basis)
    )
    
    # Short Entry Conditions (idunno2)
    cond_short = (
        (close > myma2) &
        (close > myma) &
        (close > myema) &
        (close > myema2) &
        crossover(close, upper) &
        crossover(myrsi, myrsi3) &
        crossover(close, basis)
    )
    
    # Exit Conditions
    # Original Pine had invalid boolean logic (standalone indicator calls)
    # Translated to rely on the crossover/crossunder components only
    exit_long = crossover(open_price, upper)
    exit_short = crossunder(open_price, lower)
    
    # Shift conditions by 1 to prevent lookahead (signal at i based on close i-1)
    cond_long_shift = cond_long.shift(1).fillna(False)
    cond_short_shift = cond_short.shift(1).fillna(False)
    exit_long_shift = exit_long.shift(1).fillna(False)
    exit_short_shift = exit_short.shift(1).fillna(False)
    
    # State Machine to track position
    pos = 0
    for i in range(n):
        # Prioritize exits
        if pos == 1 and exit_long_shift.iloc[i]:
            pos = 0
        elif pos == -1 and exit_short_shift.iloc[i]:
            pos = 0
        # Then entries
        elif cond_long_shift.iloc[i]:
            pos = 1
        elif cond_short_shift.iloc[i]:
            pos = -1
        
        signals[i] = pos
    
    return signals
