#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1w ADX trend filter and volume confirmation.
In ranging markets (ADX < 25), price crossing above/below Alligator lines signals mean reversion.
Volume spike confirms participation. Uses 12h timeframe to target 50-150 total trades over 4 years.
Works in bull/bear via trend filter: only trade when market is ranging (ADX < 25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams Alligator: SMoothed Moving Average (SMMA)
    # Jaw: SMMA(13, 8) of median price
    # Teeth: SMMA(8, 5) of median price
    # Lips: SMMA(5, 3) of median price
    median_price_12h = (high_12h + low_12h) / 2.0
    
    def smma(series, period):
        # Smoothed Moving Average: similar to Wilder's smoothing
        sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
        # First value is SMA, then smooth: SMMA(t) = (SMMA(t-1)*(period-1) + price(t)) / period
        smoothed = np.full_like(series, np.nan, dtype=float)
        if len(series) >= period:
            smoothed[period-1] = sma[period-1]
            for i in range(period, len(series)):
                if not np.isnan(smoothed[i-1]):
                    smoothed[i] = (smoothed[i-1] * (period-1) + series[i]) / period
        return smoothed
    
    jaw = smma(median_price_12h, 13)
    teeth = smma(median_price_12h, 8)
    lips = smma(median_price_12h, 5)
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1-week ADX (14-period)
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr0 = np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])
    tr = np.concatenate([[tr0], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(series, period):
        alpha = 1.0 / period
        smoothed = np.full_like(series, np.nan, dtype=float)
        if len(series) >= 1:
            # First value is the first non-nan
            first_idx = np.where(~np.isnan(series))[0]
            if len(first_idx) > 0:
                smoothed[first_idx[0]] = series[first_idx[0]]
                for i in range(first_idx[0]+1, len(series)):
                    if not np.isnan(series[i]):
                        smoothed[i] = alpha * series[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    tr_14 = wilders_smooth(tr, 14)
    dm_plus_14 = wilders_smooth(dm_plus, 14)
    dm_minus_14 = wilders_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilders_smooth(dx, 14)
    
    # ADX < 25 = low trend strength (good for mean reversion)
    low_trend = adx < 25
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    low_trend_aligned = align_htf_to_ltf(prices, df_1w, low_trend.astype(float))
    
    # Get 12h data for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(low_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator lines: Lips (fastest), Teeth, Jaw (slowest)
        # In ranging market: price > Lips and Lips > Teeth and Teeth > Jaw = uptrend
        #                  price < Lips and Lips < Teeth and Teeth < Jaw = downtrend
        # We trade reversals: when price crosses Lips in opposite direction of alignment
        
        lips = lips_aligned[i]
        teeth = teeth_aligned[i]
        jaw = jaw_aligned[i]
        
        # Check for Alligator alignment (all three lines ordered)
        aligned_up = lips > teeth and teeth > jaw
        aligned_down = lips < teeth and teeth < jaw
        
        # Price relative to Lips
        price_above_lips = close[i] > lips
        price_below_lips = close[i] < lips
        
        # Entry conditions: price crosses Lips in opposite direction of alignment + volume + low trend
        # When aligned up but price crosses below Lips = potential down reversal (short)
        # When aligned down but price crosses above Lips = potential up reversal (long)
        vol_confirm = vol_spike[i]
        trend_filter = low_trend_aligned[i] > 0.5
        
        short_setup = aligned_up and price_below_lips and vol_confirm and trend_filter
        long_setup = aligned_down and price_above_lips and vol_confirm and trend_filter
        
        # Exit when price crosses Lips in direction of alignment (trend resumption)
        exit_long = position == 1 and price_below_lips and aligned_down
        exit_short = position == -1 and price_above_lips and aligned_up
        
        # Execute signals
        if long_setup and position != 1:
            position = 1
            signals[i] = position_size
        elif short_setup and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_alligator_adx_volume"
timeframe = "12h"
leverage = 1.0