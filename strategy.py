#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: Camarilla pivot breakouts at R3/S3 levels with weekly trend filter and volume spikes capture strong directional moves while avoiding whipsaws. Designed for low trade frequency (target 15-35/year) on 12h timeframe to minimize fee drag in both bull and bear markets. Weekly trend ensures alignment with higher timeframe momentum, reducing counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Camarilla levels from previous 12h bar (use previous bar's high-low-close)
    # We calculate for each bar using the prior bar's range
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # fill first value
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Typical price for pivot calculation
    prev_typical = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels: R3/S3 are the most significant breakout levels
    R3 = prev_typical + range_val * 1.1 / 2.0
    S3 = prev_typical - range_val * 1.1 / 2.0
    
    # Weekly trend filter: EMA50 on weekly close
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 2.0 * 30-period average
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 30-period volume average, 50-week EMA
    start_idx = max(30, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_level = R3[i]
        s3_level = S3[i]
        ema50_w = ema50_1w_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine weekly trend: price vs weekly EMA50
            uptrend = close_val > ema50_w
            downtrend = close_val < ema50_w
            
            if uptrend and vol_conf:
                # Long: break above R3 with volume
                if close_val > r3_level:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_conf:
                # Short: break below S3 with volume
                if close_val < s3_level:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit: price re-enters below R3 or weekly trend turns down
            if close_val < r3_level or close_val < ema50_w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price re-enters above S3 or weekly trend turns up
            if close_val > s3_level or close_val > ema50_w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0