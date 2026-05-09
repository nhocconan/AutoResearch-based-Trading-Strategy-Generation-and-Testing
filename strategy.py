#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Trend_Filter_With_RSI_And_Chop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).cumsum()
    volatility_diff = np.diff(volatility, prepend=volatility[0])
    
    # Avoid division by zero
    volatility_sum = pd.Series(volatility_diff).rolling(window=er_period, min_periods=1).sum()
    change_sum = pd.Series(change).rolling(window=er_period, min_periods=1).sum()
    er = np.where(volatility_sum != 0, change_sum / volatility_sum, 0)
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) for momentum filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1d data for chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate choppy market indicator (14-period)
    atr_period = 14
    tr1 = np.abs(df_1d['high'] - df_1d['low'])
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    
    # Chop formula: 100 * log10(sum(atr) / (max(high) - min(low))) / log10(period)
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    range_hl = highest_high - lowest_low
    chop = np.where(range_hl != 0, 100 * np.log10(sum_atr / range_hl) / np.log10(14), 50)
    
    # Align KAMA, RSI, and chop to 4h
    kama_4h = align_htf_to_ltf(prices, df_1d, kama)
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi)
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    
    # Get 4h EMA20 for additional trend confirmation
    ema20_4h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 34  # Need enough data for chop calculation
    
    for i in range(start_idx, n):
        if (np.isnan(kama_4h[i]) or np.isnan(rsi_4h[i]) or 
            np.isnan(chop_4h[i]) or np.isnan(ema20_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama_4h[i]
        rsi_val = rsi_4h[i]
        chop_val = chop_4h[i]
        ema20_val = ema20_4h[i]
        
        # Chop filter: only trade when market is not too choppy (< 61.8) or not too trending (> 38.2)
        # We'll use mean reversion in choppy markets (chop > 61.8) and trend following in trending markets (chop < 38.2)
        is_choppy = chop_val > 61.8
        is_trending = chop_val < 38.2
        
        if position == 0:
            # Enter long: price above KAMA and above EMA20 with RSI > 50 in trending market
            # OR price below KAMA and below EMA20 with RSI < 50 in choppy market (mean reversion)
            if (is_trending and close[i] > kama_val and close[i] > ema20_val and rsi_val > 50) or \
               (is_choppy and close[i] < kama_val and close[i] < ema20_val and rsi_val < 50):
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA and below EMA20 with RSI < 50 in trending market
            # OR price above KAMA and above EMA20 with RSI > 50 in choppy market (mean reversion)
            elif (is_trending and close[i] < kama_val and close[i] < ema20_val and rsi_val < 50) or \
                 (is_choppy and close[i] > kama_val and close[i] > ema20_val and rsi_val > 50):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: reverse conditions
            if (is_trending and close[i] < kama_val) or (is_choppy and close[i] > kama_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: reverse conditions
            if (is_trending and close[i] > kama_val) or (is_choppy and close[i] < kama_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals