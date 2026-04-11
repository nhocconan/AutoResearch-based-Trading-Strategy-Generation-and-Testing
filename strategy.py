#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_williams_alligator_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Calculate Williams Alligator from weekly data
    # Alligator lines: Jaw (13-period SMMA, 8 bars ahead), Teeth (8-period SMMA, 5 bars ahead), Lips (5-period SMMA, 3 bars ahead)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    median_price_1w = (high_1w + low_1w) / 2
    
    # Calculate SMMA (Smoothed Moving Average)
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev_SMMA * (period-1) + Current_Value) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(median_price_1w, 13)
    teeth = smma(median_price_1w, 8)
    lips = smma(median_price_1w, 5)
    
    # Shift jaws forward by 8 bars, teeth by 5, lips by 3 (as per Alligator definition)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Invalidate the shifted values that look ahead
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Align weekly Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips_shifted)
    
    # Calculate weekly ADX for trend strength
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth TR, DM+ and DM- with Wilder's smoothing (similar to EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: SMMA = (Prev_SMMA * (period-1) + Current_Value) / period
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1w = wilders_smooth(tr, 14)
    dm_plus_smooth = wilders_smooth(dm_plus, 14)
    dm_minus_smooth = wilders_smooth(dm_minus, 14)
    
    # Calculate DI+ and DI-
    di_plus = np.where(atr_1w != 0, dm_plus_smooth / atr_1w * 100, 0)
    di_minus = np.where(atr_1w != 0, dm_minus_smooth / atr_1w * 100, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smooth(dx, 14)
    
    # Align weekly ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 60-period EMA for 6h trend filter (close > EMA = uptrend)
    close_s = pd.Series(close)
    ema_60 = close_s.ewm(span=60, adjust=False, min_periods=60).values
    
    # Volume confirmation: 6h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(ema_60[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Alligator conditions: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        alligator_short = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Trend filters: ADX > 25 for strong trend, price > EMA60 for uptrend bias
        strong_trend = adx_aligned[i] > 25
        uptrend_bias = price_close > ema_60[i]
        downtrend_bias = price_close < ema_60[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * vol_ma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Alligator aligned up + strong trend + uptrend bias + volume
        if alligator_long and strong_trend and uptrend_bias and vol_confirm:
            enter_long = True
        
        # Short: Alligator aligned down + strong trend + downtrend bias + volume
        if alligator_short and strong_trend and downtrend_bias and vol_confirm:
            enter_short = True
        
        # Exit conditions: when Alligator reverses or trend weakens
        exit_long = not (alligator_long and strong_trend and uptrend_bias)
        exit_short = not (alligator_short and strong_trend and downtrend_bias)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 6h Williams Alligator strategy with weekly trend confirmation.
# Uses weekly Williams Alligator (Lips/Teeth/Jaw) for trend direction and entry signals.
# Filters trades with weekly ADX > 25 for strong trends and 6h EMA60 for bias alignment.
# Requires volume confirmation (1.5x 20-period average) to avoid false breakouts.
# Exits when Alligator alignment breaks or trend weakens.
# Designed to work in both bull and bear markets by capturing strong trends.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Weekly timeframe provides robust trend identification, reducing whipsaws in ranging markets.