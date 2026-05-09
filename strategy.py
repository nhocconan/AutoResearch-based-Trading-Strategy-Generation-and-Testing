#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_Direction_RSI_Filter_Chop_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA(20) for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema20_1w = close_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate KAMA on daily close
    close_series = pd.Series(close)
    change = abs(close_series.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = [close[0]]
    for i in range(1, len(close)):
        kama.append(kama[-1] + sc.iloc[i] * (close[i] - kama[-1]))
    kama = np.array(kama)
    
    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate Choppiness Index(14)
    atr = []
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(abs(high - np.roll(close, 1)))
    tr3 = pd.Series(abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_vals = tr.rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    sum_atr = pd.Series(atr_vals).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr / (max_high - min_low + 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up, RSI > 50, chop < 61.8 (trending), price above weekly EMA
            if kama[i] > kama[i-1] and rsi[i] > 50 and chop[i] < 61.8 and close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI < 50, chop < 61.8 (trending), price below weekly EMA
            elif kama[i] < kama[i-1] and rsi[i] < 50 and chop[i] < 61.8 and close[i] < ema20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA reverses down OR chop > 61.8 (ranging)
            if kama[i] < kama[i-1] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA reverses up OR chop > 61.8 (ranging)
            if kama[i] > kama[i-1] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals