#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h long/short at 1d Camarilla pivot levels with 1d EMA trend filter and volume spike.
# Uses Camarilla pivot levels (S1/R1, S2/R2, S3/R3) from 1d data as support/resistance.
# Goes long when price breaks above R1 with 1d uptrend and volume spike.
# Goes short when price breaks below S1 with 1d downtrend and volume spike.
# Exits when price returns to the central pivot point (PP) or trend reverses.
# Designed to work in both bull (breakouts above R1) and bear (breakdowns below S1).
# Target: 15-30 trades/year to avoid fee drag on 12h timeframe.
name = "12h_Camarilla_Pivot_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1-day Camarilla pivot levels
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    # R2 = C + (H - L) * 1.1 / 6
    # S2 = C - (H - L) * 1.1 / 6
    # R3 = C + (H - L) * 1.1 / 4
    # S3 = C - (H - L) * 1.1 / 4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_pp = (high_1d + low_1d + close_1d) / 3.0
    pivot_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    pivot_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d Camarilla levels and EMA to 12h timeframe (use previous day's values)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pivot_pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, pivot_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, pivot_s1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 30-period EMA (moderate threshold for 12h)
    vol_ema30 = pd.Series(volume).ewm(span=30, adjust=False, min_periods=30).mean().values
    vol_confirm = volume > (1.5 * vol_ema30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least 1 day of data for pivots/EMA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ema30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above R1 + 1d uptrend + volume spike
            if (price > r1_aligned[i] and price > ema_34_1d_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 + 1d downtrend + volume spike
            elif (price < s1_aligned[i] and price < ema_34_1d_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to pivot PP or trend reverses
            if price < pp_aligned[i] or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot PP or trend reverses
            if price > pp_aligned[i] or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals