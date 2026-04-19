# 4h_Camarilla_R1S1_Breakout_Volume_Trend_Filter
# Hypothesis: 4h Camarilla R1/S1 breakout with volume confirmation and EMA trend filter
# Uses 1d Camarilla levels (statistically significant support/resistance) and 4h EMA for trend direction
# Volume confirmation ensures institutional participation. Designed for 4h to target 75-200 total trades over 4 years.
# Works in bull/bear via EMA trend filter - only long in uptrend, short in downtrend.

name = "4h_Camarilla_R1S1_Breakout_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0

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
    
    # EMA(34) for trend filter on 4h data
    def ema(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        multiplier = 2.0 / (period + 1)
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]) and not np.isnan(arr[i]):
                result[i] = arr[i] * multiplier + result[i-1] * (1 - multiplier)
            else:
                result[i] = np.nan
        return result
    
    # 4h data for EMA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 35:
        return np.zeros(n)
    
    ema_34_4h = ema(df_4h['close'].values, 34)
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Previous day's Camarilla levels (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    ph = df_1d['high'].shift(1).values  # Previous day high
    pl = df_1d['low'].shift(1).values   # Previous day low
    pc = df_1d['close'].shift(1).values # Previous day close
    
    # Handle first value where shift creates NaN
    ph[0] = ph[1] if len(ph) > 1 else ph[0]
    pl[0] = pl[1] if len(pl) > 1 else pl[0]
    pc[0] = pc[1] if len(pc) > 1 else pc[0]
    
    # Camarilla calculations
    rang = ph - pl
    # Avoid division by zero or negative ranges
    rang = np.where(rang <= 0, np.full_like(rang, 1e-10), rang)
    
    r1 = pc + (rang * 1.1 / 12)
    s1 = pc - (rang * 1.1 / 12)
    r4 = pc + (rang * 1.1 / 2)
    s4 = pc - (rang * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: EMA direction
        uptrend = close[i] > ema_34_4h_aligned[i]
        downtrend = close[i] < ema_34_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and uptrend
            if (close[i] > r1_aligned[i] and 
                volume_confirm[i] and 
                uptrend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and downtrend
            elif (close[i] < s1_aligned[i] and 
                  volume_confirm[i] and 
                  downtrend):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 or trend reverses
            if (close[i] < s1_aligned[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 or trend reverses
            if (close[i] > r1_aligned[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals