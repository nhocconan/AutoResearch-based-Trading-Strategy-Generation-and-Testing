#!/usr/bin/env python3
# Hypothesis: 6h Williams %R reversal with 12h EMA50 trend filter and volume spike
# Long when Williams %R crosses above -20 from below (oversold reversal) with EMA50 uptrend and volume > 1.5x average
# Short when Williams %R crosses below -80 from above (overbought reversal) with EMA50 downtrend and volume > 1.5x average
# Exit when Williams %R returns to opposite extreme (-80 for long, -20 for short) or trend reverses
# Williams %R identifies exhaustion points, EMA50 filters trend direction, volume confirms conviction
# Designed to capture mean reversals in both trending and ranging markets with controlled frequency
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "6h_WilliamsR_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Williams %R (14 period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    highest_high = pd.Series(df_12h['high']).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(df_12h['low']).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - df_12h['close']) / (highest_high - lowest_low)
    williams_r = williams_r.replace(0, np.nan).values  # avoid division by zero
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(williams_r_aligned[i-1]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R crosses above -20 from below, EMA50 uptrend, volume spike
            if (williams_r_aligned[i] > -20 and williams_r_aligned[i-1] <= -20 and
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses below -80 from above, EMA50 downtrend, volume spike
            elif (williams_r_aligned[i] < -80 and williams_r_aligned[i-1] >= -80 and
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to -80 or trend reverses
            if (williams_r_aligned[i] < -80) or (ema50_12h_aligned[i] < ema50_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to -20 or trend reverses
            if (williams_r_aligned[i] > -20) or (ema50_12h_aligned[i] > ema50_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals