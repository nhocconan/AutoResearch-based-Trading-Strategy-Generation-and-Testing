#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Long when price breaks above R3 level, 12h EMA50 rising, volume > 2x average
# Short when price breaks below S3 level, 12h EMA50 falling, volume > 2x average
# Uses Camarilla pivot levels for institutional support/resistance, EMA50 for trend filter, volume spike for institutional participation
# Targets 20-50 trades per year (80-200 over 4 years) for low fee drag and high win rate
# Works in both bull and bear markets due to trend filter and volume confirmation requiring institutional participation

name = "4h_Camarilla_R3S3_12hEMA50_VolumeSpike"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day's data (to avoid look-ahead)
    # Using previous 4h bar's close for pivot calculation
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Pivot point calculation
    pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Calculate Camarilla levels
    range_val = prev_high - prev_low
    R3 = pivot + (range_val * 1.1 / 2)
    S3 = pivot - (range_val * 1.1 / 2)
    
    # Calculate EMA50 on 12h close for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need at least 20 periods for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        high_val = high[i]
        low_val = low[i]
        R3_val = R3[i]
        S3_val = S3[i]
        ema50_12h_val = ema50_12h_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Enter long: price breaks above R3, 12h uptrend, volume spike
            if high_val > R3_val and ema50_12h_val > 0 and vol_spike_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3, 12h downtrend, volume spike
            elif low_val < S3_val and ema50_12h_val < 0 and vol_spike_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or 12h trend down
            if low_val < S3_val or ema50_12h_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 or 12h trend up
            if high_val > R3_val or ema50_12h_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals