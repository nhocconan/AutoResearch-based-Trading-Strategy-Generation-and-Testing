#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla Pivot (R1/S1) Breakout with 1-day Trend Filter and Volume Confirmation
# Uses daily Camarilla pivot levels (R1/S1) for breakout entries
# Long when price breaks above R1 + daily EMA(50) uptrend + volume spike
# Short when price breaks below S1 + daily EMA(50) downtrend + volume spike
# Camarilla levels provide statistically significant support/resistance
# Daily trend filter ensures alignment with higher timeframe momentum
# Volume spike confirms institutional participation in breakout
# Targets 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (R1, S1) from previous day
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    camarilla_r1 = daily_close + (daily_high - daily_low) * 1.1 / 12
    camarilla_s1 = daily_close - (daily_high - daily_low) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (available after daily bar closes)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate daily EMA(50) for trend filter
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        ema50_1d_val = ema50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Break above R1 + daily uptrend + volume spike
            if close[i] > r1 and close[i] > ema50_1d_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Break below S1 + daily downtrend + volume spike
            elif close[i] < s1 and close[i] < ema50_1d_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price falls below S1 OR daily trend turns down
            if close[i] < s1 or close[i] < ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price rises above R1 OR daily trend turns up
            if close[i] > r1 or close[i] > ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals