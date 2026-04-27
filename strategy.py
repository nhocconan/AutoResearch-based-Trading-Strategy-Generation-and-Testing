#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Volume_And_Chop_Filter
Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) to determine trend direction on 1d timeframe.
Enter long when price > KAMA AND volume > 1.5x 20-period average AND choppiness index < 50 (trending market).
Enter short when price < KAMA AND volume > 1.5x 20-period average AND choppiness index < 50.
Exit when price crosses back over KAMA or volume drops below average.
Designed for 1d timeframe to achieve 30-100 total trades over 4 years (7-25/year).
KAMA adapts to market noise, reducing false signals in ranging markets when combined with chop filter.
Volume confirmation ensures trades occur with participation. Works in both bull and bear markets by following adaptive trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA and choppiness index
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d KAMA ( Kaufman Adaptive Moving Average )
    close_1d = pd.Series(df_1d['close'].values)
    # Efficiency Ratio: |net change| / sum of absolute changes over 10 periods
    change = abs(close_1d.diff(10))
    volatility = close_1d.diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)  # Avoid division by zero
    # Smoothing constants: fastest EMA=2, slowest EMA=30
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # ER*(0.4667) + 0.0667 squared
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[9] = close_1d.iloc[9]  # Start after 10 periods
    for i in range(10, len(close_1d)):
        if not np.isnan(sc.iloc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close_1d.iloc[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_values = kama.values
    
    # Calculate 1d Choppiness Index
    # True Range = max(high-low, abs(high-previous close), abs(low-previous close))
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).sum()  # ATR(14) as sum of TR
    # Chop = 100 * log10( sum(tr14) / (atr(14) * 14) ) / log10(14)
    sum_tr14 = tr.rolling(14).sum()
    chop = 100 * np.log10(sum_tr14 / (atr * 14)) / np.log10(14)
    chop_values = chop.values
    
    # Volume average for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 1d timeframe (same timeframe, no alignment needed for same TF)
    # But we still use align_htf_to_ltf for safety and consistency with rules
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_values)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need KAMA (10), Chop (14), Vol avg (20)
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        chop_val = chop_aligned[i]
        vol_val = volume[i]
        vol_avg_val = vol_avg[i]
        
        if position == 0:
            # Look for entry: price vs KAMA with volume confirmation AND chop filter (trending market)
            # Long: price > KAMA AND volume > 1.5x avg AND chop < 50 (trending)
            long_condition = (close_val > kama_val) and (vol_val > 1.5 * vol_avg_val) and (chop_val < 50)
            # Short: price < KAMA AND volume > 1.5x avg AND chop < 50 (trending)
            short_condition = (close_val < kama_val) and (vol_val > 1.5 * vol_avg_val) and (chop_val < 50)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price crosses below KAMA OR volume drops below average
            exit_condition = (close_val < kama_val) or (vol_val < vol_avg_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price crosses above KAMA OR volume drops below average
            exit_condition = (close_val > kama_val) or (vol_val < vol_avg_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_With_Volume_And_Chop_Filter"
timeframe = "1d"
leverage = 1.0