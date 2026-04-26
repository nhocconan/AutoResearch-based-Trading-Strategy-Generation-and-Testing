#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolumeSpike
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and 1d volume spike confirmation.
Trade only in direction of 4h trend. Uses discrete sizing 0.20 to target 15-35 trades/year on 1h.
Designed for both bull and bear markets via 4h trend filter and 1d volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    close_4h_series = pd.Series(close_4h)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume median for spike detection
    volume_1d = df_1d['volume'].values
    volume_1d_series = pd.Series(volume_1d)
    vol_median_20_1d = volume_1d_series.rolling(window=20, min_periods=20).median().values
    vol_median_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20_1d)
    
    # Calculate Camarilla levels from prior 1h bar (using rolling window)
    # We need prior bar's high/low/close for Camarilla calculation
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # Shift by 1 to get prior bar values
    prior_high = high_series.shift(1).values
    prior_low = low_series.shift(1).values
    prior_close = close_series.shift(1).values
    
    # Calculate Camarilla R3, S3, R4, S4 for each prior 1h bar
    rng = prior_high - prior_low
    r3 = prior_close + 1.125 * rng
    s3 = prior_close - 1.125 * rng
    r4 = prior_close + 1.5 * rng
    s4 = prior_close - 1.5 * rng
    
    # Volume spike: 1h volume > 1.5x 20-period median 1d volume (scaled)
    # Scale 1d median volume to approximate 1h equivalent (divide by 6 for 6x 1h bars in 1d)
    vol_median_20_1h_approx = vol_median_20_1d_aligned / 6.0
    volume_spike = volume > (1.5 * vol_median_20_1h_approx)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for 4h EMA, 20 for 1d volume median, 1 for prior bar
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(vol_median_20_1d_aligned[i]) or
            np.isnan(r3[i]) or
            np.isnan(s3[i]) or
            np.isnan(r4[i]) or
            np.isnan(s4[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_4h_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price > R3 and volume spike, in uptrend (close > EMA50)
            long_entry = (close_val > r3[i]) and vol_spike and (close_val > ema_50_val)
            # Short: price < S3 and volume spike, in downtrend (close < EMA50)
            short_entry = (close_val < s3[i]) and vol_spike and (close_val < ema_50_val)
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal or at R4 (take profit)
            if close_val < ema_50_val or close_val > r4[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or at S4 (take profit)
            if close_val > ema_50_val or close_val < s4[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolumeSpike"
timeframe = "1h"
leverage = 1.0