#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike
Hypothesis: On 4h timeframe, enter long when price breaks above Camarilla R3 level AND 12h trend is up (close > EMA50) AND volume > 2.0x 20-period average. Enter short when price breaks below S3 level AND 12h trend is down (close < EMA50) AND volume spike. Uses Camarilla pivot levels for precise support/resistance, 12h EMA50 for higher timeframe trend alignment, and volume confirmation for institutional participation. Designed for moderate trade frequency (20-40/year) to balance opportunity and fee drag while capturing strong trends in both bull and bear markets.
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h_series = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Daily Camarilla Pivot Levels (R3, S3)
    # Based on previous day's OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3 = close + ((high-low)*1.1/4), S3 = close - ((high-low)*1.1/4)
    camarilla_r3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup (50), volume MA warmup (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions relative to Camarilla levels
        breakout_above_r3 = close[i] > camarilla_r3_aligned[i]
        breakout_below_s3 = close[i] < camarilla_s3_aligned[i]
        
        # 12h trend filter
        trend_uptrend = close[i] > ema_50_12h_aligned[i]
        trend_downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: price above R3 + 12h uptrend + volume spike
            long_signal = breakout_above_r3 and trend_uptrend and volume_spike[i]
            
            # Short: price below S3 + 12h downtrend + volume spike
            short_signal = breakout_below_s3 and trend_downtrend and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below S3 OR trend change to downtrend
            if close[i] < camarilla_s3_aligned[i] or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R3 OR trend change to uptrend
            if close[i] > camarilla_r3_aligned[i] or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0