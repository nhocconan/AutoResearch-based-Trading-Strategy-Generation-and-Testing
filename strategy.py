#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation.
    # Uses 12h volume spike to confirm breakouts from 1d Camarilla levels.
    # Camarilla pivots provide intraday support/resistance; volume confirms institutional participation.
    # Works in bull/bear via volatility expansion on breakouts.
    # Target: 50-150 total trades over 4 years = 12-37/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data for volume confirmation (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivots (based on previous day's range)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's high-low-close
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    range_1d = high_1d - low_1d
    camarilla_r4 = close_1d + 1.5 * range_1d
    camarilla_r3 = close_1d + 1.1 * range_1d
    camarilla_s3 = close_1d - 1.1 * range_1d
    camarilla_s4 = close_1d - 1.5 * range_1d
    
    # Calculate 12h volume ratio (current volume / 20-period average)
    vol_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio = vol_12h / np.where(vol_ma_20 == 0, 1, vol_ma_20)  # Avoid division by zero
    
    # Align HTF indicators to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmation = vol_ratio_aligned[i] > 1.5
        
        # Breakout conditions at Camarilla extremes
        breakout_long = close[i] > camarilla_r4_aligned[i]  # Break above R4
        breakout_short = close[i] < camarilla_s4_aligned[i]  # Break below S4
        
        # Fade conditions at Camarilla mid-levels (counter-trend)
        fade_long = close[i] < camarilla_s3_aligned[i] and close[i] > camarilla_s4_aligned[i]  # Between S3-S4
        fade_short = close[i] > camarilla_r3_aligned[i] and close[i] < camarilla_r4_aligned[i]  # Between R3-R4
        
        # Entry logic: breakouts with volume confirmation, fades without (mean reversion)
        long_entry = (breakout_long and volume_confirmation) or (fade_long and not volume_confirmation)
        short_entry = (breakout_short and volume_confirmation) or (fade_short and not volume_confirmation)
        
        # Exit conditions: return to opposite Camarilla level
        long_exit = close[i] < camarilla_s3_aligned[i]
        short_exit = close[i] > camarilla_r3_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_1d_camarilla_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0