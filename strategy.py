#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels (R3/S3) with volume spike and 4h EMA50 trend filter
# Long when price breaks above 1d Camarilla R3 AND 4h EMA50 rising AND volume > 2.0 * 20-period avg volume
# Short when price breaks below 1d Camarilla S3 AND 4h EMA50 falling AND volume > 2.0 * 20-period avg volume
# Exit when price crosses back through the 1d Camarilla midpoint (Pivot point)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Camarilla levels work well in ranging markets and provide clear breakout/breakdown points
# EMA50 filter ensures we trade with the intermediate trend, reducing whipsaw
# Volume spike confirms institutional participation in the move
# Effective in both bull markets (buying strength) and bear markets (selling weakness)

name = "4h_1dCamarillaR3S3_EMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least one completed daily bar
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day's range)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # We use R3 and S3 for breakouts, and Pivot (PP) for exit
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First bar uses current bar (no look-ahead)
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Calculate pivot point and Camarilla levels
    pp = (prev_high + prev_low + prev_close) / 3.0
    r3 = pp + (prev_high - prev_low) * 1.1 / 4.0
    s3 = pp - (prev_high - prev_low) * 1.1 / 4.0
    
    # Align 1d Camarilla levels to 4h timeframe (wait for completed daily bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Calculate 4h EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_rising = ema50 > np.roll(ema50, 1)  # Rising if current > previous
    ema50_falling = ema50 < np.roll(ema50, 1)  # Falling if current < previous
    ema50_rising[0] = False
    ema50_falling[0] = False
    
    # Volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(ema50[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R3, EMA50 rising, volume spike, in session
            if (close[i] > r3_aligned[i] and ema50_rising[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S3, EMA50 falling, volume spike, in session
            elif (close[i] < s3_aligned[i] and ema50_falling[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 1d Camarilla pivot point
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 1d Camarilla pivot point
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals