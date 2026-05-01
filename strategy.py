#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h EHLERS FISHER TRANSFORM + 1d VOLUME REGIME + 1w PIVOT DIRECTION
# Fisher Transform identifies extreme price reversals with leading signals.
# Volume regime filter (high/low volume) confirms participation.
# 1w pivot direction provides structural bias to avoid counter-trend extremes in strong trends.
# Long: Fisher < -1.5 AND volume regime = high AND price > 1w pivot point
# Short: Fisher > +1.5 AND volume regime = high AND price < 1w pivot point
# Works in bull/bear by aligning reversals with higher timeframe structure and volume confirmation.
# Target: 15-30 trades/year (60-120 total over 4 years) with discrete sizing 0.25.

name = "6h_EhlersFisher_1dVolumeRegime_1wPivotDir_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for volume regime and Fisher
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # === 1d Ehlers Fisher Transform (period=10) ===
    hl2_1d = (df_1d['high'].values + df_1d['low'].values) / 2.0
    max_hl2 = pd.Series(hl2_1d).rolling(window=10, min_periods=10).max().values
    min_hl2 = pd.Series(hl2_1d).rolling(window=10, min_periods=10).min().values
    range_hl2 = max_hl2 - min_hl2
    # Avoid division by zero
    range_hl2 = np.where(range_hl2 == 0, 1e-10, range_hl2)
    value_1d = 0.66 * ((hl2_1d - min_hl2) / range_hl2 - 0.5) + 0.67 * np.roll(0.66 * ((hl2_1d - min_hl2) / range_hl2 - 0.5) + 0.67 * np.roll(np.zeros_like(hl2_1d), 1), 1)
    # Initialize first value
    value_1d[0] = 0
    # Calculate Fisher Transform recursively
    fish_1d = np.zeros_like(hl2_1d)
    for i in range(1, len(hl2_1d)):
        fish_1d[i] = 0.5 * np.log((1 + value_1d[i]) / (1 - value_1d[i] + 1e-10)) + 0.5 * fish_1d[i-1]
    # Align Fisher to 6h
    fish_1d_aligned = align_htf_to_ltf(prices, df_1d, fish_1d)
    
    # === 1d Volume Regime (High/Low) ===
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.where(vol_ma_1d > 0, volume / vol_ma_1d, 1.0)
    # Volume regime: 1 = high volume (>1.5x MA), 0 = normal/low
    vol_regime_1d = (vol_ratio_1d > 1.5).astype(float)
    vol_regime_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_regime_1d)
    
    # === 1w Camarilla Pivot Points (using prior week OHLC) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Prior week OHLC for current week's pivot
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot_1w = (prev_high + prev_low + prev_close) / 3.0
    range_1w = prev_high - prev_low
    r1_1w = prev_close + (range_1w * 1.1 / 12)
    s1_1w = prev_close - (range_1w * 1.1 / 12)
    r3_1w = prev_close + (range_1w * 1.1 / 4)
    s3_1w = prev_close - (range_1w * 1.1 / 4)
    
    # Align 1w levels to 6h
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for indicators
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(fish_1d_aligned[i]) or np.isnan(vol_regime_1d_aligned[i]) or
            np.isnan(pivot_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_fish = fish_1d_aligned[i]
        curr_vol_regime = vol_regime_1d_aligned[i]
        curr_pivot = pivot_1w_aligned[i]
        curr_r3 = r3_1w_aligned[i]
        curr_s3 = s3_1w_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Fisher < -1.5 (extreme low) AND high volume AND price above weekly pivot
            if (curr_fish < -1.5 and 
                curr_vol_regime > 0.5 and 
                curr_close > curr_pivot):
                signals[i] = 0.25
                position = 1
            # Short: Fisher > +1.5 (extreme high) AND high volume AND price below weekly pivot
            elif (curr_fish > 1.5 and 
                  curr_vol_regime > 0.5 and 
                  curr_close < curr_pivot):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Fisher > 0 (reversal signal) OR volume drops low OR price breaks below weekly S1
            if (curr_fish > 0 or 
                curr_vol_regime < 0.5 or 
                curr_close < s1_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Fisher < 0 (reversal signal) OR volume drops low OR price breaks above weekly R1
            if (curr_fish < 0 or 
                curr_vol_regime < 0.5 or 
                curr_close > r1_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals