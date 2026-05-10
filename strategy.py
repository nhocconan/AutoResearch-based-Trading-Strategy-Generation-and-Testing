#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hEMA34_Trend_VolumeS
Hypothesis: On 4h timeframe, enter long when price breaks above Camarilla R3 level and short when price breaks below S3 level, filtered by 12h EMA34 trend direction and volume spike. Exit when price returns to Camarilla Pivot level. This structure captures institutional reversal points with trend alignment, limiting trades to 20-40/year to avoid fee drag. Works in bull/bear via trend filter and volume confirmation.
"""

name = "4h_Camarilla_R3_S3_Breakout_12hEMA34_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # 4h data for Camarilla levels and price
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels (based on previous day's OHLC)
    # We'll calculate daily OHLC first
    # Since we're on 4h timeframe, we need to group by day
    # Simpler approach: use rolling window of 6 bars (since 6*4h = 24h)
    # But better to use actual daily data from 1d timeframe
    # We already have df_1d, so we can use its open, high, low, close
    # For each 4h bar, we use the previous day's OHLC
    
    # Extract daily OHLC from df_1d
    # We need to align the previous day's values to each 4h bar
    # Since df_1d is already aligned to 4h via align_htf_to_ltf, we can use shifted values
    
    # Get aligned daily OHLC
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Align them to 4h timeframe
    open_1d_aligned = align_htf_to_ltf(prices, df_1d, open_1d)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate Camarilla levels using previous day's OHLC
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    # Pivot = (high + low + close) / 3
    
    # Shift by 1 to use previous day's values
    prev_high = np.roll(high_1d_aligned, 1)
    prev_low = np.roll(low_1d_aligned, 1)
    prev_close = np.roll(close_1d_aligned, 1)
    
    # First value will be invalid due to roll, we'll handle it
    prev_high[0] = 0
    prev_low[0] = 0
    prev_close[0] = 0
    
    # Calculate Camarilla levels
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    Pivot = (prev_high + prev_low + prev_close) / 3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for calculations
    start_idx = 1  # since we rolled by 1
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN or invalid
        if (np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(vol_ma20_1d_aligned[i]) or
            np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(Pivot[i]) or
            prev_high[i] == 0 or prev_low[i] == 0 or prev_close[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs 12h EMA34
        uptrend_12h = close[i] > ema34_12h_aligned[i]
        downtrend_12h = close[i] < ema34_12h_aligned[i]
        
        # Volume filter: current 4h volume > 2x 1d 20-period MA
        volume_filter = volume[i] > vol_ma20_1d_aligned[i] * 2.0
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long: price breaks above R3 in uptrend with volume
            if close[i] > R3[i] and uptrend_12h and volume_filter and in_session:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 in downtrend with volume
            elif close[i] < S3[i] and downtrend_12h and volume_filter and in_session:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to Pivot level or trend fails
            if close[i] <= Pivot[i] or not uptrend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to Pivot level or trend fails
            if close[i] >= Pivot[i] or not downtrend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals