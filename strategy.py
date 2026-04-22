#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with RSI filter and volume confirmation. 
# KAMA adapts to market noise - effective in both trending and ranging markets.
# Long when KAMA turns up + RSI > 50 + volume > 1.5x average
# Short when KAMA turns down + RSI < 50 + volume > 1.5x average
# Exit when KAMA reverses direction
# Target: 10-25 trades/year to minimize fee drag on 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data (same as primary timeframe for calculations)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = |Change| / Volatility
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)
    # Avoid division by zero
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators (though 1d data on 1d timeframe, alignment ensures proper handling)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        kama = kama_aligned[i]
        rsi = rsi_aligned[i]
        vol_ma = vol_ma_aligned[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_filter = volume > 1.5 * vol_ma
        
        if position == 0:
            # Long: KAMA turning up + RSI > 50 + volume filter
            if i > 0 and kama > kama_aligned[i-1] and rsi > 50 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down + RSI < 50 + volume filter
            elif i > 0 and kama < kama_aligned[i-1] and rsi < 50 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: KAMA reverses direction
            if (position == 1 and kama < kama_aligned[i-1]) or \
               (position == -1 and kama > kama_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_RSI_Volume_Filter"
timeframe = "1d"
leverage = 1.0