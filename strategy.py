#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Camarilla pivot levels with volume confirmation and 12h trend filter.
# Long when price breaks above R1 with volume in a 12h uptrend.
# Short when price breaks below S1 with volume in a 12h downtrend.
# Exit when price retests the pivot level (PP).
# Uses volume spike (2x 20-period EMA) to confirm breakout strength.
# Designed for low trade frequency (20-40/year) to minimize fee drag and capture strong moves.

name = "4h_Camarilla_R1S1_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from prior day
    pp = np.zeros_like(close_1d)      # Pivot Point
    r1 = np.zeros_like(close_1d)      # Resistance 1
    s1 = np.zeros_like(close_1d)      # Support 1
    
    for i in range(1, len(close_1d)):
        # Prior day's high, low, close
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        # Camarilla equations
        pp[i] = (ph + pl + pc) / 3
        r1[i] = pc + 1.1 * (ph - pl) / 12
        s1[i] = pc - 1.1 * (ph - pl) / 12
    
    # First day has no prior data
    pp[0] = r1[0] = s1[0] = np.nan
    
    # Align Camarilla levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 12h trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_12h_up = ema_34_12h[1:] > ema_34_12h[:-1]  # Rising 12h EMA
    trend_12h_up = np.concatenate([[False], trend_12h_up])  # Align with 12h index
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up.astype(float))
    
    # Volume confirmation: current volume > 2x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for volume EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(trend_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: break above R1 with volume in 12h uptrend
            if (trend_12h_aligned[i] > 0.5 and          # 12h uptrend
                close[i] > r1_aligned[i] and            # Break above R1
                vol_confirm[i]):                        # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short setup: break below S1 with volume in 12h downtrend
            elif (trend_12h_aligned[i] <= 0.5 and       # 12h downtrend
                  close[i] < s1_aligned[i] and          # Break below S1
                  vol_confirm[i]):                      # Volume confirmation
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: retest pivot point (PP)
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: retest pivot point (PP)
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals