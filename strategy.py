#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX(14) + Williams Alligator combination for trend strength and direction.
# Uses 12h ADX to filter for trending markets (ADX > 25) and avoid chop.
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) from 6h data defines trend direction:
#   Bullish: Lips > Teeth > Jaw (all rising)
#   Bearish: Jaw > Teeth > Lips (all falling)
# Entry on pullback to Teeth (8-period SMMA) in direction of trend with volume confirmation (>1.5x 20-bar avg).
# Exits when price crosses Jaw (13-period SMMA) or ADX falls below 20 (trend weakening).
# Discrete position sizing at ±0.25 to balance return and fee drag.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag on 6h timeframe.
# Works in bull markets via trend continuation and in bear markets via short trends.

name = "6h_ADX_WilliamsAlligator_Trend_Pullback_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 12h data ONCE before loop for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ADX(14) for trend strength filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original length
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_14 != 0, 100 * dm_plus_14 / tr_14, 0)
    di_minus = np.where(tr_14 != 0, 100 * dm_minus_14 / tr_14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.full_like(dx, np.nan)
    adx_period = 14
    if len(dx) >= adx_period:
        # First ADX value: average of first adx_period DX values
        adx[adx_period-1] = np.nanmean(dx[:adx_period])
        # Subsequent values: Wilder's smoothing of DX
        for i in range(adx_period, len(dx)):
            adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period
    
    # Align 12h ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Williams Alligator on 6h data: SMMA (Smoothed Moving Average) = Wilder's smoothing
    def smma(data, period):
        return wilders_smoothing(data, period)
    
    # Jaw (13-period SMMA of median price)
    median_price = (high + low) / 2
    jaw = smma(median_price, 13)
    # Teeth (8-period SMMA of median price)
    teeth = smma(median_price, 8)
    # Lips (5-period SMMA of median price)
    lips = smma(median_price, 5)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_adx = adx_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend strength filter: ADX > 25 for trending market
        is_trending = curr_adx > 25
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator alignment: Lips > Teeth > Jaw (all rising)
            bullish_alignment = (curr_lips > curr_teeth > curr_jaw)
            # Bearish Alligator alignment: Jaw > Teeth > Lips (all falling)
            bearish_alignment = (curr_jaw > curr_teeth > curr_lips)
            
            # Long entry: bullish trend, pullback to Teeth, volume confirmation
            if (is_trending and bullish_alignment and 
                curr_close >= curr_teeth * 0.998 and curr_close <= curr_teeth * 1.002 and  # near Teeth
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish trend, pullback to Teeth, volume confirmation
            elif (is_trending and bearish_alignment and 
                  curr_close >= curr_teeth * 0.998 and curr_close <= curr_teeth * 1.002 and  # near Teeth
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit conditions:
            # 1. Price crosses below Jaw (trend reversal)
            # 2. ADX falls below 20 (trend weakening)
            if (curr_close < curr_jaw or curr_adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price crosses above Jaw (trend reversal)
            # 2. ADX falls below 20 (trend weakening)
            if (curr_close > curr_jaw or curr_adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals