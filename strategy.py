#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator (Jaw/Teeth/Lips) with 1d volume spike and 1d ADX trend filter
# Long when Lips > Teeth > Jaw (bullish alignment) AND volume > 2.0x 20-period average AND 1d ADX > 25
# Short when Lips < Teeth < Jaw (bearish alignment) AND volume > 2.0x 20-period average AND 1d ADX > 25
# Exit when Alligator alignment breaks OR 1d ADX < 20 (trend weakening)
# Uses discrete sizing (0.25) to limit fee drag. Target: 20-50 trades/year per symbol.
# Williams Alligator identifies trend phases via smoothed medians, volume confirms participation,
# 1d ADX filters for strong trends to avoid choppy markets. Works in bull/bear via directional alignment.

name = "4h_WilliamsAlligator_1dADX_Trend_VolumeSpike"
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
    
    # Get 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 4h data (using close prices)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current Price) / Period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Shift as per Alligator definition
    jaw = np.concatenate([np.full(8, np.nan), jaw[:-8]]) if len(jaw) > 8 else np.full_like(jaw, np.nan)
    teeth = np.concatenate([np.full(5, np.nan), teeth[:-5]]) if len(teeth) > 5 else np.full_like(teeth, np.nan)
    lips = np.concatenate([np.full(3, np.nan), lips[:-3]]) if len(lips) > 3 else np.full_like(lips, np.nan)
    
    # Bullish alignment: Lips > Teeth > Jaw
    bullish_align = (lips > teeth) & (teeth > jaw)
    # Bearish alignment: Lips < Teeth < Jaw
    bearish_align = (lips < teeth) & (teeth < jaw)
    
    # Calculate 1d ADX for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+, DM- (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(arr, period):
        """Wilder's smoothing (EMA with alpha=1/period)"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is average of first 'period' values
        result[period-1] = np.nanmean(arr[:period])
        # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current Price) / Period
        for i in range(period, len(arr)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    period_adx = 14
    tr_sm = wilders_smoothing(tr, period_adx)
    dm_plus_sm = wilders_smoothing(dm_plus, period_adx)
    dm_minus_sm = wilders_smoothing(dm_minus, period_adx)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_sm / tr_sm
    di_minus = 100 * dm_minus_sm / tr_sm
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilders_smoothing(dx, period_adx)
    
    # Trend filters: ADX > 25 for strong trend, ADX < 20 for weakening trend
    strong_trend = adx > 25
    weakening_trend = adx < 20
    
    # Align 1d indicators to 4h timeframe
    strong_trend_aligned = align_htf_to_ltf(prices, df_1d, strong_trend.astype(float))
    weakening_trend_aligned = align_htf_to_ltf(prices, df_1d, weakening_trend.astype(float))
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(strong_trend_aligned[i]) or np.isnan(weakening_trend_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: bullish Alligator alignment AND volume spike AND strong 1d trend
            if (bullish_align[i] and 
                volume_filter[i] and 
                strong_trend_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish Alligator alignment AND volume spike AND strong 1d trend
            elif (bearish_align[i] and 
                  volume_filter[i] and 
                  strong_trend_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish alignment OR weakening 1d trend
            if (bearish_align[i] or 
                weakening_trend_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish alignment OR weakening 1d trend
            if (bullish_align[i] or 
                weakening_trend_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals