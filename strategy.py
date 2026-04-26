#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeFilter
Hypothesis: On 1h timeframe, use Camarilla R3/S3 from 4h pivot points for breakout entries with 4h trend filter (close > 4h EMA21) and volume confirmation (>1.8x 20-period average). 1h is used only for entry timing precision; 4h provides signal direction. This reduces trade frequency while capturing strong intraday moves in both bull and bear markets. Target: 20-50 trades/year.
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
    if len(df_4h) < 21:  # Need at least 21 periods for EMA21
        return np.zeros(n)
    
    # Calculate 4h OHLC for Camarilla pivot points
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels (based on previous 4h bar's range)
    # Camarilla R3 = close + 1.1*(high - low)/2
    # Camarilla S3 = close - 1.1*(high - low)/2
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    
    # Set first value to NaN (no previous bar)
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    prev_close_4h[0] = np.nan
    
    camarilla_r3 = prev_close_4h + 1.1 * (prev_high_4h - prev_low_4h) / 2
    camarilla_s3 = prev_close_4h - 1.1 * (prev_high_4h - prev_low_4h) / 2
    
    # Calculate 4h EMA21 for trend filter
    close_4h_series = pd.Series(close_4h)
    ema_21_4h = close_4h_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align Camarilla levels and EMA to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Volume confirmation: volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Camarilla (4h EMA21) + volume MA warmup
    start_idx = max(21, 20)  # 21 for EMA21
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_21_4h_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # 4h trend alignment
        trend_4h_uptrend = close[i] > ema_21_4h_aligned[i]
        trend_4h_downtrend = close[i] < ema_21_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 + 4h uptrend + volume spike
            long_signal = (close[i] > camarilla_r3_aligned[i]) and trend_4h_uptrend and volume_spike[i]
            
            # Short: price breaks below S3 + 4h downtrend + volume spike
            short_signal = (close[i] < camarilla_s3_aligned[i]) and trend_4h_downtrend and volume_spike[i]
            
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
            # Exit: price touches S3 OR 4h trend turns down
            if (close[i] < camarilla_s3_aligned[i] or not trend_4h_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price touches R3 OR 4h trend turns up
            if (close[i] > camarilla_r3_aligned[i] or not trend_4h_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeFilter"
timeframe = "1h"
leverage = 1.0