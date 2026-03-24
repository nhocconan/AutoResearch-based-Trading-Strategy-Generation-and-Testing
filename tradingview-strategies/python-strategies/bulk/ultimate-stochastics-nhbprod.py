#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "Ultimate Stochastics Strategy by NHBprod"
timeframe = "4h"
leverage = 1

# Strategy Parameters (Defaults from Pine Script)
TP_PERCENT = 0.14
SL_PERCENT = 0.08
FAST_K_PERIOD = 9
SLOW_K_PERIOD = 18
SLOW_D_PERIOD = 4
STOCH_OVERBOUGHT = 60
STOCH_OVERSOLD = 90
STOCH_SMOOTHING = 'SMA'
INPUT_SINCE_STOCH = 1
USE_STOCH = True
STOCH_EXIT = True
USE_OPPOSITE = True
USE_TRADE_MANAGEMENT = True
POSITION_TYPE = 'Long & Short'

def sma(series, length):
    return series.rolling(window=length).mean()

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def wma(series, length):
    weights = np.arange(1, length + 1)
    def calc_wma(x):
        return np.dot(x, weights) / weights.sum()
    return series.rolling(window=length).apply(calc_wma, raw=True)

def rma(series, length):
    alpha = 1 / length
    return series.ewm(alpha=alpha, adjust=False).mean()

def hull_ma(series, length):
    half = int(length / 2)
    sqrt_len = int(np.round(np.sqrt(length)))
    if half < 1: half = 1
    if sqrt_len < 1: sqrt_len = 1
    wma_half = wma(series, half)
    wma_full = wma(series, length)
    hull_input = 2 * wma_half - wma_full
    return wma(hull_input, sqrt_len)

def get_ma(series, length, method):
    if method == 'SMA': return sma(series, length)
    if method == 'EMA': return ema(series, length)
    if method == 'WMA': return wma(series, length)
    if method == 'RMA': return rma(series, length)
    if method == 'HMA': return hull_ma(series, length)
    return sma(series, length)

def calculate_stoch(high, low, close, k_period, smooth_k_period, smooth_d_period, smooth_method):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denominator = highest_high - lowest_low
    raw_k = np.where(denominator == 0, 0, 100 * (close - lowest_low) / denominator)
    raw_k = pd.Series(raw_k, index=close.index)
    
    slow_k = get_ma(raw_k, smooth_k_period, smooth_method)
    slow_d = get_ma(slow_k, smooth_d_period, smooth_method)
    return slow_k, slow_d

def generate_signals(prices):
    df = prices.copy()
    n = len(df)
    signals = np.zeros(n, dtype=int)
    
    if n == 0:
        return signals
    
    slow_k, slow_d = calculate_stoch(df['high'], df['low'], df['close'], 
                                     FAST_K_PERIOD, SLOW_K_PERIOD, SLOW_D_PERIOD, STOCH_SMOOTHING)
    
    crossover = (slow_k > slow_d) & (slow_k.shift(1) <= slow_d.shift(1))
    crossunder = (slow_k < slow_d) & (slow_k.shift(1) >= slow_d.shift(1))
    
    buy_signal_raw = crossover & (slow_k < STOCH_OVERSOLD)
    sell_signal_raw = crossunder & (slow_k > STOCH_OVERBOUGHT)
    
    buy_signal_raw = buy_signal_raw.fillna(False)
    sell_signal_raw = sell_signal_raw.fillna(False)
    
    is_buy_signal = np.zeros(n, dtype=bool)
    is_sell_signal = np.zeros(n, dtype=bool)
    
    buy_indices = np.where(buy_signal_raw.values)[0]
    sell_indices = np.where(sell_signal_raw.values)[0]
    
    for idx in buy_indices:
        end = min(idx + INPUT_SINCE_STOCH + 1, n)
        is_buy_signal[idx:end] = True
        
    for idx in sell_indices:
        end = min(idx + INPUT_SINCE_STOCH + 1, n)
        is_sell_signal[idx:end] = True
        
    master_buy = is_buy_signal & USE_STOCH
    master_sell = is_sell_signal & STOCH_EXIT
    
    position = 0
    entry_price = 0.0
    
    close_vals = df['close'].values
    high_vals = df['high'].values
    low_vals = df['low'].values
    
    for i in range(n):
        if position == 1:
            if USE_TRADE_MANAGEMENT:
                tp_level = entry_price * (1 + TP_PERCENT)
                sl_level = entry_price * (1 - SL_PERCENT)
                if high_vals[i] >= tp_level or low_vals[i] <= sl_level:
                    position = 0
                    entry_price = 0.0
            if position == 1 and USE_OPPOSITE and master_sell[i]:
                position = 0
                entry_price = 0.0
                
        elif position == -1:
            if USE_TRADE_MANAGEMENT:
                tp_level = entry_price * (1 - TP_PERCENT)
                sl_level = entry_price * (1 + SL_PERCENT)
                if low_vals[i] <= tp_level or high_vals[i] >= sl_level:
                    position = 0
                    entry_price = 0.0
            if position == -1 and USE_OPPOSITE and master_buy[i]:
                position = 0
                entry_price = 0.0
        
        if position == 0:
            if POSITION_TYPE in ['Long', 'Long & Short'] and master_buy[i]:
                position = 1
                entry_price = close_vals[i]
            elif POSITION_TYPE in ['Short', 'Long & Short'] and master_sell[i]:
                position = -1
                entry_price = close_vals[i]
        
        signals[i] = position
        
    return signals