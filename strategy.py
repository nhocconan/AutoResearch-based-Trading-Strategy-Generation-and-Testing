#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R3/S3 breakout with volume spike and weekly ADX trend filter.
Long when price breaks above Camarilla R3 level AND volume > 2.0x 20-period average AND weekly ADX > 25 (trending market).
Short when price breaks below Camarilla S3 level AND volume > 2.0x 20-period average AND weekly ADX > 25 (trending market).
Exit when price reverts to Camarilla pivot (central level).
Uses 1d for price/volume/Camarilla levels, weekly for ADX trend filter to ensure we trade only in trending conditions.
Targets 30-100 total trades over 4 years (7-25/year). Camarilla levels provide high-probability reversal/breakout points,
volume confirmation reduces fakeouts, weekly ADX ensures we avoid choppy markets where breakouts fail.
Works in bull markets (captures uptrend breakouts) and bear markets (captures downtrend breakdowns).
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
    
    # Get 1d data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels on 1d timeframe
    # Pivot = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # Resistance levels: R3 = C + (H-L) * 1.1/2
    r3 = close_1d + range_1d * 1.1 / 2.0
    # Support levels: S3 = C - (H-L) * 1.1/2
    s3 = close_1d - range_1d * 1.1 / 2.0
    # Central pivot (exit level)
    central_pivot = pivot
    
    # Calculate volume average (20-period) on 1d
    volume_series = pd.Series(volume_1d)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX on weekly timeframe (14-period)
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to 0 to avoid lookback issue (will be filtered by min_periods anyway)
    tr[0] = 0
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    # Set first values to 0
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(values, period):
        """Wilder's smoothing: first value is SMA, then recursive"""
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            # First value is simple average
            result[period-1] = np.mean(values[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(values)):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    period = 14
    atr_1w = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # Directional Indicators
    di_plus = np.where(atr_1w != 0, dm_plus_smooth / atr_1w * 100, 0)
    di_minus = np.where(atr_1w != 0, dm_minus_smooth / atr_1w * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1w = wilders_smoothing(dx, period)
    
    # Align 1d Camarilla levels, volume MA, and weekly ADX to 1d timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    central_pivot_aligned = align_htf_to_ltf(prices, df_1d, central_pivot)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(central_pivot_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        pivot_level = central_pivot_aligned[i]
        vol_ma = volume_ma_aligned[i]
        adx = adx_1w_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Camarilla R3 AND volume > 2.0x avg AND weekly ADX > 25 (strong trend)
            if price > r3_level and vol > 2.0 * vol_ma and adx > 25:
                signals[i] = 0.25
                position = 1
            # Short: price < Camarilla S3 AND volume > 2.0x avg AND weekly ADX > 25 (strong trend)
            elif price < s3_level and vol > 2.0 * vol_ma and adx > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Camarilla pivot (central level)
            if price < pivot_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Camarilla pivot (central level)
            if price > pivot_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_CamarillaR3S3_Volume_WeeklyADX_Filter"
timeframe = "1d"
leverage = 1.0