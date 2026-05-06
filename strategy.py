#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d ADX(25) trend filter and volume spike
# Williams Alligator: Jaw (13-period smoothed median), Teeth (8-period), Lips (5-period)
# Trend signal when all three lines are aligned (Jaw > Teeth > Lips for long, reverse for short)
# 1d ADX(25) confirms trend strength, volume spike (>1.8x 20-bar avg) confirms momentum
# Designed to work in both bull/bear markets by capturing strong trends with confirmation filters
# Target: 80-160 total trades over 4 years (20-40/year) to avoid fee drag

name = "4h_WilliamsAlligator_1dADX25_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 35:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator on 1d timeframe
    # Median price = (high + low) / 2
    median_price = (high_1d + low_1d) / 2
    
    # Jaw: 13-period SMMA of median, shifted 8 bars
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.concatenate([np.full(8, np.nan), jaw_raw[:-8]]) if len(jaw_raw) > 8 else np.full_like(jaw_raw, np.nan)
    
    # Teeth: 8-period SMMA of median, shifted 5 bars
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.concatenate([np.full(5, np.nan), teeth_raw[:-5]]) if len(teeth_raw) > 5 else np.full_like(teeth_raw, np.nan)
    
    # Lips: 5-period SMMA of median, shifted 3 bars
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.concatenate([np.full(3, np.nan), lips_raw[:-3]]) if len(lips_raw) > 3 else np.full_like(lips_raw, np.nan)
    
    # Calculate 1d ADX(25) trend filter
    # TR = max(high-low, |high-prev_close|, |low-prev_close|)
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, +DM, -DM with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    atr_1d = wilder_smooth(tr, 25)
    dm_plus_smooth = wilder_smooth(dm_plus, 25)
    dm_minus_smooth = wilder_smooth(dm_minus, 25)
    
    # DI+ = 100 * smoothed +DM / ATR, DI- = 100 * smoothed -DM / ATR
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX = 100 * |DI+ - DI-| / (DI+ + DI-)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    # ADX = smoothed DX
    adx_1d = wilder_smooth(dx, 25)
    
    # Calculate ATR(14) for 4h timeframe (for stoploss)
    tr1_4h = np.abs(high[1:] - low[1:])
    tr2_4h = np.abs(high[1:] - close[:-1])
    tr3_4h = np.abs(low[1:] - close[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter (>1.8x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_20)
    
    # Align HTF indicators to 4h timeframe (primary)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0
    short_extreme = 0.0
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(atr_4h[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        if position == 0:
            # Long signal: Jaw > Teeth > Lips (bullish alignment) AND strong trend (ADX > 25) AND volume spike
            if jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i] and adx_1d_aligned[i] > 25 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            # Short signal: Jaw < Teeth < Lips (bearish alignment) AND strong trend (ADX > 25) AND volume spike
            elif jaw_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i] and adx_1d_aligned[i] > 25 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, close[i])
            # Exit long: price retraces 25% of ATR from extreme
            if close[i] <= long_extreme - 0.25 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, close[i])
            # Exit short: price retraces 25% of ATR from extreme
            if close[i] >= short_extreme + 0.25 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals