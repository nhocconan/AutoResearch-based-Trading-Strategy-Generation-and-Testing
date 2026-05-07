#!/usr/bin/env python3
# 4h_Camarilla_R3S3_Breakout_1dTrend_VolumeFilter
# Hypothesis: Price action at Camarilla R3/S3 levels on 4h chart, filtered by 1d trend (EMA34) and volume spikes.
# Long when price breaks above R3 with volume and 1d EMA34 uptrend.
# Short when price breaks below S3 with volume and 1d EMA34 downtrend.
# Uses Camarilla pivot levels for high-probability reversal/breakout points in ranging and trending markets.
# Volume filter reduces false breakouts. Trend filter ensures alignment with higher timeframe momentum.
# Designed for low trade frequency (~20-40/year) to minimize fee drag and work in both bull and bear markets.
timeframe = "4h"
name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeFilter"
leverage = 1.0

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
    
    # Calculate Camarilla levels for each bar using prior day's OHLC
    # Note: For 4h chart, we calculate Camarilla based on prior 1d candle
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    # Need at least one full day of data
    for i in range(1, n):
        # Get prior day's OHLC (assuming 24*60/4 = 24 bars per day on 4h chart)
        # We'll use a simpler approach: calculate once per day using prior day's values
        pass  # Will implement properly below
    
    # Instead, calculate Camarilla levels from prior day's OHLC using vectorized approach
    # Resample logic is handled by getting actual 1d data and aligning
    
    # Get 1d OHLC data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: 
    # R4 = close + (high-low)*1.1/2
    # R3 = close + (high-low)*1.1/4
    # S3 = close - (high-low)*1.1/4
    # S4 = close - (high-low)*1.1/2
    range_1d = high_1d - low_1d
    camarilla_r3_1d = close_1d + range_1d * 1.1 / 4
    camarilla_s3_1d = close_1d - range_1d * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (each 1d bar's levels apply to following 24 bars)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0 * 24-period average (1 day average on 4h chart)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index 24 to ensure we have volume MA
    start_idx = max(24, 34)  # Ensure we have enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 with volume and 1d uptrend
            if close[i] > camarilla_r3_aligned[i] and volume[i] > 2.0 * vol_ma[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 with volume and 1d downtrend
            elif close[i] < camarilla_s3_aligned[i] and volume[i] > 2.0 * vol_ma[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below Camarilla S3 (mean reversion) or trend fails
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above Camarilla R3 (mean reversion) or trend fails
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals