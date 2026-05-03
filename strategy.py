#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d volume confirmation and 1w trend filter
# Bollinger Band squeeze identifies low volatility periods primed for breakout.
# Breakout direction confirmed by 1d volume spike and aligned with 1week EMA50 trend.
# Designed for low trade frequency (12-37/year) on 6h timeframe to minimize fee drag.
# Works in both bull and bear markets by trading volatility contractions/expansions within higher timeframe trend.

name = "6h_BollingerSqueeze_1dVolumeSpike_1wEMA50_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d and 1w data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d Bollinger Bands (20, 2.0)
    close_1d = pd.Series(df_1d['close'].values)
    bb_mid = close_1d.ewm(span=20, adjust=False, min_periods=20).mean().values
    bb_std = close_1d.ewm(span=20, adjust=False, min_periods=20).std().values
    bb_upper = bb_mid + (2.0 * bb_std)
    bb_lower = bb_mid - (2.0 * bb_std)
    bb_width = (bb_upper - bb_lower) / bb_mid  # Normalized width
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 6h timeframe
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Align 1w EMA50 to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 6h Bollinger Band squeeze (20, 2.0)
    close_6h = pd.Series(close)
    bb_mid_6h = close_6h.ewm(span=20, adjust=False, min_periods=20).mean().values
    bb_std_6h = close_6h.ewm(span=20, adjust=False, min_periods=20).std().values
    bb_upper_6h = bb_mid_6h + (2.0 * bb_std_6h)
    bb_lower_6h = bb_mid_6h - (2.0 * bb_std_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(bb_width_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(bb_upper_6h[i]) or np.isnan(bb_lower_6h[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend direction
        is_uptrend = close[i] > ema_50_1w_aligned[i]
        is_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: Bollinger Band squeeze breakout above upper band with volume spike in uptrend
            if (close[i] > bb_upper_6h[i] and 
                bb_width_aligned[i] < 0.05 and  # Squeeze threshold: BB width < 5%
                volume_spike_aligned[i] and 
                is_uptrend):
                signals[i] = 0.25
                position = 1
            # Short: Bollinger Band squeeze breakout below lower band with volume spike in downtrend
            elif (close[i] < bb_lower_6h[i] and 
                  bb_width_aligned[i] < 0.05 and  # Squeeze threshold: BB width < 5%
                  volume_spike_aligned[i] and 
                  is_downtrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle band or squeeze re-establishes
            if close[i] <= bb_mid_6h[i] or bb_width_aligned[i] < 0.03:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle band or squeeze re-establishes
            if close[i] >= bb_mid_6h[i] or bb_width_aligned[i] < 0.03:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals