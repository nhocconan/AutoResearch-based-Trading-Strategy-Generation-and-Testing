#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: On 1h timeframe, use Camarilla R1/S1 from 4h pivot points for breakout entries with 4h trend filter (close > 4h EMA20) and volume confirmation (>1.5x 24-period average). Target: 15-35 trades/year by using tight 1h entries only in alignment with 4h trend and volume spikes. Uses R1/S1 (vs R2/S2) for more frequent but still filtered breakouts in both bull and bear regimes via 4h trend/volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculation and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need at least 20 periods for EMA20
        return np.zeros(n)
    
    # Calculate 4h OHLC for Camarilla pivot points
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels R1/S1 (based on previous 4h bar's range)
    # Camarilla R1 = close + 1.1*(high - low)/12
    # Camarilla S1 = close - 1.1*(high - low)/12
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    
    # Set first value to NaN (no previous bar)
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    prev_close_4h[0] = np.nan
    
    camarilla_r1 = prev_close_4h + 1.1 * (prev_high_4h - prev_low_4h) / 12
    camarilla_s1 = prev_close_4h - 1.1 * (prev_high_4h - prev_low_4h) / 12
    
    # Calculate 4h EMA20 for trend filter
    close_4h_series = pd.Series(close_4h)
    ema_20_4h = close_4h_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align Camarilla levels and EMA to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume confirmation: volume > 1.5x 24-period average (4h equivalent: 6*4=24)
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Camarilla (4h EMA20) + volume MA
    start_idx = max(20, 24)  # 20 for EMA20, 24 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # 4h trend alignment
        trend_4h_uptrend = close[i] > ema_20_4h_aligned[i]
        trend_4h_downtrend = close[i] < ema_20_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + 4h uptrend + volume spike
            # Require confirmation: price outside bands for 2 consecutive bars
            long_breakout = (close[i] > camarilla_r1_aligned[i]) and (close[i-1] > camarilla_r1_aligned[i-1])
            long_signal = long_breakout and trend_4h_uptrend and volume_spike[i]
            
            # Short: price breaks below S1 + 4h downtrend + volume spike
            short_breakout = (close[i] < camarilla_s1_aligned[i]) and (close[i-1] < camarilla_s1_aligned[i-1])
            short_signal = short_breakout and trend_4h_downtrend and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price touches S1 OR 4h trend turns down
            if (close[i] < camarilla_s1_aligned[i] or not trend_4h_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price touches R1 OR 4h trend turns up
            if (close[i] > camarilla_r1_aligned[i] or not trend_4h_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0