#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels with breakout/fade logic
# Fade at R3/S3 levels (mean reversion in ranging markets)
# Breakout continuation at R4/S4 levels (trend following in strong moves)
# Uses 1d Camarilla calculations for structure, 6h for execution timing
# Designed to work in both bull/bear markets via adaptive logic
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_1d_camarilla_breakout_fade_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # Camarilla: HLC3 = (high + low + close) / 3
    # Range = high - low
    hlc3 = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d
    
    # Camarilla levels for current day based on previous day's action
    # We shift by 1 to avoid look-ahead (use previous day's data)
    hlc3_prev = np.roll(hlc3, 1)
    rng_prev = np.roll(rng, 1)
    hlc3_prev[0] = np.nan
    rng_prev[0] = np.nan
    
    camarilla_h4 = hlc3_prev + rng_prev * 1.1 / 2  # R4
    camarilla_l4 = hlc3_prev - rng_prev * 1.1 / 2  # S4
    camarilla_h3 = hlc3_prev + rng_prev * 1.1 / 4  # R3
    camarilla_l3 = hlc3_prev - rng_prev * 1.1 / 4  # S3
    camarilla_h2 = hlc3_prev + rng_prev * 1.1 / 6  # R2
    camarilla_l2 = hlc3_prev - rng_prev * 1.1 / 6  # S2
    camarilla_h1 = hlc3_prev + rng_prev * 1.1 / 12 # R1
    camarilla_l1 = hlc3_prev - rng_prev * 1.1 / 12 # S1
    camarilla_pivot = hlc3_prev                    # Pivot point
    
    # Align 1d Camarilla levels to 6h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    p_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Calculate 6h average volume (20-period) for confirmation
    vol_s = pd.Series(volume)
    avg_vol_20 = vol_s.rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC) - avoid low liquidity periods
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(avg_vol_20[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.3x average 6h volume (20-period)
        volume_confirmed = volume[i] > 1.3 * avg_vol_20[i] if not np.isnan(avg_vol_20[i]) else False
        
        if position == 1:  # Long position
            # Exit long if price falls below S3 (mean reversion failure)
            if close[i] < l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price rises above R3 (mean reversion failure)
            if close[i] > h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Fade at R3/S3 levels (mean reversion)
            if close[i] > h3_aligned[i] and close[i] < h4_aligned[i] and volume_confirmed:
                # Price between R3 and R4 - fade short expecting reversion to pivot
                position = -1
                signals[i] = -0.25
            elif close[i] < l3_aligned[i] and close[i] > l4_aligned[i] and volume_confirmed:
                # Price between S3 and S4 - fade long expecting reversion to pivot
                position = 1
                signals[i] = 0.25
            # Breakout continuation at R4/S4 levels (trend following)
            elif close[i] > h4_aligned[i] and volume_confirmed:
                # Price breaks above R4 - go long expecting continuation
                position = 1
                signals[i] = 0.25
            elif close[i] < l4_aligned[i] and volume_confirmed:
                # Price breaks below S4 - go short expecting continuation
                position = -1
                signals[i] = -0.25
    
    return signals