#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirm
Hypothesis: On 12h timeframe, price breaking above/below Camarilla R3/S3 levels with daily trend filter (EMA34) and volume confirmation captures strong momentum moves. Uses discrete sizing (0.30) to limit fee drag. Works in bull/bear by requiring alignment with daily trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 34:  # Need 34 for daily EMA34
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous 12h bar
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range / 2.0
    s3 = prev_close - 1.1 * camarilla_range / 2.0
    
    # Volume confirmation: volume > 1.5x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 1.5)
    
    # Load daily data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    bars_since_entry = 0
    
    # Start after warmup (need 1 for previous bar, 34 for EMA)
    start_idx = 34
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        close_val = close[i]
        r3_val = r3[i]
        s3_val = s3[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Long logic: price breaks above R3 with volume spike and daily uptrend
        long_condition = close_val > r3_val and vol_spike and (close_val > ema_34_val)
        # Short logic: price breaks below S3 with volume spike and daily downtrend
        short_condition = close_val < s3_val and vol_spike and (close_val < ema_34_val)
        
        # Exit logic: price re-enters Camarilla H3/L3 or daily trend reversal
        # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
        h3 = prev_close[i] + 1.1 * camarilla_range[i] / 4.0
        l3 = prev_close[i] - 1.1 * camarilla_range[i] / 4.0
        exit_long = close_val < h3 or close_val < ema_34_val
        exit_short = close_val > l3 or close_val > ema_34_val
        
        # Minimum holding period: 2 bars (12h * 2 = 1 day)
        if position != 0 and bars_since_entry < 2:
            # Hold position regardless of signals
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            bars_since_entry = 0
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            bars_since_entry = 0
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0