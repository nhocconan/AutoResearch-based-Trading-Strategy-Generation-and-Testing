# [EXPERIMENT #65400] 4h_Weekly_Pivot_R1S1_Breakout_Volume_Trend
# Hypothesis: Weekly pivot point R1/S1 breakouts with volume confirmation and trend filter
# Weekly pivots provide stronger support/resistance than daily, reducing false breakouts
# Volume ensures institutional participation, trend filter avoids chop
# Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year)
# Works in bull/bear via weekly pivot structure and volume confirmation

name = "4h_Weekly_Pivot_R1S1_Breakout_Volume_Trend"
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
    
    # Weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    ph = df_1w['high'].shift(1).values  # Previous week high
    pl = df_1w['low'].shift(1).values   # Previous week low
    pc = df_1w['close'].shift(1).values # Previous week close
    
    # Weekly pivot calculations (standard pivot point formula)
    pivot = (ph + pl + pc) / 3.0
    r1 = 2 * pivot - pl
    s1 = 2 * pivot - ph
    r2 = pivot + (ph - pl)
    s2 = pivot - (ph - pl)
    r3 = ph + 2 * (pivot - pl)
    s3 = pl - 2 * (ph - pivot)
    
    # Align weekly pivot levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Volume confirmation: volume > 1.8 * 30-period average (stricter for fewer trades)
    volume_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (volume_ma * 1.8)
    
    # Trend filter: 50-period EMA on 4h data
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    # Price above EMA = uptrend, below EMA = downtrend
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50)  # Ensure enough data for EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(ema_50[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend based on EMA
        uptrend = close[i] > ema_50[i]
        downtrend = close[i] < ema_50[i]
        
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