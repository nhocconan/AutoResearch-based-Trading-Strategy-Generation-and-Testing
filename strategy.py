#1/root/candidates/2025-06-20_04-46-49.py
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI + chop filter
# KAMA (Kaufman Adaptive Moving Average) adapts to market noise - slow in ranging markets, fast in trends.
# Combined with RSI for momentum and Choppiness Index for regime detection.
# Works in bull markets (trend following) and bear markets (mean reversion in chop).
# Target: 20-60 total trades over 4 years (5-15/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Load 1w data for Choppiness Index (regime filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate KAMA (10-period ER, 2 and 30 SC)
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, n=10))
    abs_change = np.sum(np.abs(np.diff(close_1d)), axis=0)
    # Manual calculation for ER
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        if i >= 10:
            change_val = np.abs(close_1d[i] - close_1d[i-10])
            abs_change_val = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
            if abs_change_val > 0:
                er[i] = change_val / abs_change_val
            else:
                er[i] = 0
    er[0:10] = 0
    
    # Smoothing Constants
    sc = (er * (0.0645 - 0.0612) + 0.0612) ** 2
    sc[0:10] = 0.0612 ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI (14-period) on 1d
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    
    # First average
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    # Subsequent averages
    for i in range(15, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[0:14] = 50  # Neutral before enough data
    
    # Calculate Choppiness Index (14-period) on 1w
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of TR over 14 periods
    tr_sum = np.zeros_like(close_1w)
    for i in range(14, len(close_1w)):
        tr_sum[i] = np.sum(tr[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    max_high = np.zeros_like(close_1w)
    min_low = np.zeros_like(close_1w)
    for i in range(14, len(close_1w)):
        max_high[i] = np.max(high_1w[i-13:i+1])
        min_low[i] = np.min(low_1w[i-13:i+1])
    
    # Chop calculation
    chop = np.zeros_like(close_1w)
    for i in range(14, len(close_1w)):
        if max_high[i] - min_low[i] > 0:
            chop[i] = 100 * np.log10(tr_sum[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50
    chop[0:14] = 50
    
    # Align indicators to 1d timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            continue
            
        # Long: Price above KAMA + RSI > 50 + Chop < 61.8 (trending market)
        if (close[i] > kama_aligned[i] and
            rsi_aligned[i] > 50 and
            chop_aligned[i] < 61.8 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short: Price below KAMA + RSI < 50 + Chop < 61.8 (trending market)
        elif (close[i] < kama_aligned[i] and
              rsi_aligned[i] < 50 and
              chop_aligned[i] < 61.8 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Chop > 61.8 (ranging market) or reverse signal
        elif position == 1 and (chop_aligned[i] > 61.8 or close[i] < kama_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (chop_aligned[i] > 61.8 or close[i] > kama_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0