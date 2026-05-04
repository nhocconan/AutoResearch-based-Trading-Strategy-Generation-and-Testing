#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d Regime Filter
# Elder Ray measures bull/bear power vs EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Regime filter: 1d ADX > 25 = trending (use Elder Ray for continuation), ADX < 20 = ranging (fade extremes)
# Long: Bull Power > 0 AND (ADX > 25 AND Bull Power increasing) OR (ADX < 20 AND Bull Power > 0.5 * ATR)
# Short: Bear Power < 0 AND (ADX > 25 AND Bear Power decreasing) OR (ADX < 20 AND Bear Power < -0.5 * ATR)
# Uses 6h for Elder Ray calculation, 1d for regime (ADX) to reduce whipsaw in sideways markets.
# Works in bull markets via longs in bullish regime and bear markets via shorts in bearish regime.
# Volume confirmation added to reduce false signals.

name = "6h_ElderRay_1dRegime_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF regime filter (ADX) - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    period = 14
    atr_1d = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, period)
    
    # Regime: ADX > 25 = trending, ADX < 20 = ranging
    adx_trending = adx > 25
    adx_ranging = adx < 20
    
    # Align 1d regime to 6h timeframe
    adx_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_trending.astype(float))
    adx_ranging_aligned = align_htf_to_ltf(prices, df_1d, adx_ranging.astype(float))
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 6h ATR for volume/filter scaling
    tr_6h1 = np.abs(high[1:] - low[1:])
    tr_6h2 = np.abs(high[1:] - close[:-1])
    tr_6h3 = np.abs(low[1:] - close[:-1])
    tr_6h = np.maximum(tr_6h1, np.maximum(tr_6h2, tr_6h3))
    tr_6h = np.concatenate([[np.nan], tr_6h])
    atr_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: 20-period volume EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_trending_aligned[i]) or np.isnan(adx_ranging_aligned[i]) or
            np.isnan(ema_13[i]) or np.isnan(atr_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions
            long_trending = (bull_power[i] > 0 and 
                           adx_trending_aligned[i] > 0.5 and 
                           i > 30 and bull_power[i] > bull_power[i-1])  # Bull power increasing
            long_ranging = (bull_power[i] > 0.5 * atr_6h[i] and 
                          adx_ranging_aligned[i] > 0.5)
            
            # Short conditions
            short_trending = (bear_power[i] < 0 and 
                            adx_trending_aligned[i] > 0.5 and 
                            i > 30 and bear_power[i] < bear_power[i-1])  # Bear power decreasing
            short_ranging = (bear_power[i] < -0.5 * atr_6h[i] and 
                           adx_ranging_aligned[i] > 0.5)
            
            if (long_trending or long_ranging) and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif (short_trending or short_ranging) and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bear power becomes positive OR regime turns strongly ranging against trend
            if (bear_power[i] > 0 or 
                (adx_ranging_aligned[i] > 0.5 and bull_power[i] < 0)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bull power becomes negative OR regime turns strongly ranging against trend
            if (bull_power[i] < 0 or 
                (adx_ranging_aligned[i] > 0.5 and bear_power[i] > 0)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals