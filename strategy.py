#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout on 1d with 1w EMA50 trend filter and volume spike confirmation.
Long when price breaks above R3 AND above 1w EMA50 AND volume > 1.5x 20-period average.
Short when price breaks below S3 AND below 1w EMA50 AND volume > 1.5x 20-period average.
Exit on opposite Camarilla level touch (R3 for long, S3 for short) or volume dry-up.
Uses discrete sizing (0.25) to minimize fee drag. Target: 20-60 trades over 4 years.
Works in bull/bear via 1w trend filter and volatility-adjusted breakouts.
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
    
    # Camarilla levels (based on previous day's range)
    # R4 = close + 1.5*(high-low), R3 = close + 1.25*(high-low), etc.
    # We calculate for previous bar to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # first bar: use current
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    range_ = prev_high - prev_low
    r3 = prev_close + 1.25 * range_
    s3 = prev_close - 1.25 * range_
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Load 1w data for HTF trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 50-period for EMA and 20 for volume MA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price breaks above R3 + above 1w EMA50 + volume spike
        long_condition = (close[i] > r3[i]) and (close[i] > ema_50_1w_aligned[i]) and volume_spike[i]
        # Short logic: price breaks below S3 + below 1w EMA50 + volume spike
        short_condition = (close[i] < s3[i]) and (close[i] < ema_50_1w_aligned[i]) and volume_spike[i]
        
        # Exit logic: touch opposite Camarilla level OR volume dry-up (below average)
        volume_dry = volume[i] < vol_ma[i]  # volume below 20-period average
        long_exit = (position == 1) and ((close[i] < s3[i]) or volume_dry)
        short_exit = (position == -1) and ((close[i] > r3[i]) or volume_dry)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0