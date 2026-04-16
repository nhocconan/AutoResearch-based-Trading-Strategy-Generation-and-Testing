#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI momentum + 1w volatility regime filter
# Long when KAMA trending up AND RSI > 50 AND 1w volatility low (below 50th percentile)
# Short when KAMA trending down AND RSI < 50 AND 1w volatility low
# Uses 1d timeframe with 1w volatility filter to reduce whipsaw in choppy markets
# Target: 20-60 total trades over 4 years (5-15/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d KAMA (adaptive moving average) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Vectorized calculation of ER
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === 1d RSI (14-period) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = 50  # Initialize first values
    
    # === 1w Volatility (ATR percentile rank) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Calculate percentile rank of current ATR (50-period lookback)
    atr_percentile = np.zeros_like(atr_1w)
    for i in range(50, len(atr_1w)):
        if i >= 50:
            window = atr_1w[i-50:i]
            atr_percentile[i] = (np.sum(window <= atr_1w[i]) / len(window)) * 100
    
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1w, atr_percentile)
    
    # Align KAMA and RSI to lower timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)  # Using 1w index for alignment
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(atr_percentile_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_percentile = atr_percentile_aligned[i]
        
        # Volatility filter: only trade when volatility is low (below 50th percentile)
        low_volatility = vol_percentile < 50
        
        # KAMA trend: price above/below KAMA
        kama_up = price > kama_val
        kama_down = price < kama_val
        
        # RSI momentum
        rsi_bullish = rsi_val > 50
        rsi_bearish = rsi_val < 50
        
        # Entry conditions
        if kama_up and rsi_bullish and low_volatility:
            signals[i] = 0.25
        elif kama_down and rsi_bearish and low_volatility:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_1wVolatilityPercentile"
timeframe = "1d"
leverage = 1.0