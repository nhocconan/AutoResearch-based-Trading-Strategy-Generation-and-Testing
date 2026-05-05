#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) + 1d ADX regime filter + volume confirmation
# Long when: Bull Power > 0 (close > EMA13) AND Bear Power < 0 AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period MA
# Short when: Bear Power < 0 (close < EMA13) AND Bull Power < 0 AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period MA
# Exit when: Bull/Bear Power crosses zero OR 1d ADX < 20 (range) OR volume drops
# Uses Elder Ray for trend strength, 1d ADX for regime filter, volume for conviction
# Timeframe: 6h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_ElderRay_1dADX_VolumeConfirm"
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
    
    # Calculate EMA13 for Elder Ray on 6h
    if len(close) >= 13:
        ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    else:
        ema13 = np.full(n, np.nan)
    
    # Elder Ray: Bull Power = Close - EMA13, Bear Power = Close - EMA13 (same calculation, interpreted differently)
    bull_power = close - ema13
    bear_power = close - ema13  # Same value, but we interpret negative as bearish
    
    # Volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on 1d timeframe (14-period)
    if len(high_1d) >= 14 and len(low_1d) >= 14 and len(close_1d) >= 14:
        # True Range
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr1[0] = 0  # First value has no previous close
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                           np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
        dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                            np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values (using Wilder's smoothing = EMA with alpha=1/period)
        def wilders_smoothing(data, period):
            if len(data) < period:
                return np.full(len(data), np.nan)
            alpha = 1.0 / period
            result = np.full(len(data), np.nan)
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        atr = wilders_smoothing(tr, 14)
        dm_plus_smooth = wilders_smoothing(dm_plus, 14)
        dm_minus_smooth = wilders_smoothing(dm_minus, 14)
        
        # Directional Indicators
        di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
        di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        adx = wilders_smoothing(dx, 14)
        
        # Regime filters
        adx_trending = adx > 25
        adx_ranging = adx < 20
    else:
        adx = np.full(len(close_1d), np.nan)
        adx_trending = np.full(len(close_1d), False)
        adx_ranging = np.full(len(close_1d), False)
    
    # Align 1d ADX regime to 6h timeframe
    adx_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_trending.astype(float))
    adx_ranging_aligned = align_htf_to_ltf(prices, df_1d, adx_ranging.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(adx_trending_aligned[i]) or np.isnan(adx_ranging_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND Bear Power < 0 (redundant but clear) AND 1d ADX trending AND volume filter
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                adx_trending_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND Bull Power < 0 (redundant but clear) AND 1d ADX trending AND volume filter
            elif (bear_power[i] < 0 and 
                  bull_power[i] < 0 and 
                  adx_trending_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR 1d ADX turns ranging OR volume drops
            if (bull_power[i] <= 0 or adx_ranging_aligned[i] == 1.0 or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 OR 1d ADX turns ranging OR volume drops
            if (bear_power[i] >= 0 or adx_ranging_aligned[i] == 1.0 or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals