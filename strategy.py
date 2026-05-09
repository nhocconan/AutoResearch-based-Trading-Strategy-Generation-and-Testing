#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ADX_Alligator_Trend_Filter"
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
    
    # Get 1d data for Alligator and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Alligator: SMAs with offsets (Williams Alligator)
    # Jaw: 13-period SMMA, 8 bars ahead
    # Teeth: 8-period SMMA, 5 bars ahead
    # Lips: 5-period SMMA, 3 bars ahead
    close_1d = df_1d['close'].values
    
    def smma(arr, period):
        # Smoothed Moving Average (SMMA)
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Williams Alligator signals: 
    # Bullish: Lips > Teeth > Jaw (green alignment)
    # Bearish: Jaw > Teeth > Lips (red alignment)
    bullish_alignment = (lips > teeth) & (teeth > jaw)
    bearish_alignment = (jaw > teeth) & (teeth > lips)
    
    # Align Alligator signals to 6h
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_alignment.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_alignment.astype(float))
    
    # ADX on 1d for trend strength
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed TR, DM+
        atr = np.zeros_like(tr)
        dm_plus_smooth = np.zeros_like(dm_plus)
        dm_minus_smooth = np.zeros_like(dm_minus)
        
        # First values: simple average
        atr[period-1] = np.mean(tr[:period])
        dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
        dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
        
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Avoid division by zero
        dx = np.full_like(tr, np.nan, dtype=float)
        valid = (dm_plus_smooth + dm_minus_smooth) != 0
        dx[valid] = 100 * np.abs(dm_plus_smooth[valid] - dm_minus_smooth[valid]) / (dm_plus_smooth[valid] + dm_minus_smooth[valid])
        
        # ADX: smoothed DX
        adx = np.full_like(tr, np.nan, dtype=float)
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])  # First ADX value
        for i in range(2*period-1, len(tr)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume filter: current 6h volume > 1.3 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Need enough data for ADX and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish = bullish_aligned[i] > 0.5
        bearish = bearish_aligned[i] > 0.5
        adx = adx_aligned[i]
        vol_filter = volume_filter[i]
        
        # Only trade when ADX > 25 (strong trend)
        strong_trend = adx > 25
        
        if position == 0:
            # Enter long: bullish Alligator alignment + strong trend + volume
            if bullish and strong_trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish Alligator alignment + strong trend + volume
            elif bearish and strong_trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish alignment or ADX < 20 (weakening trend)
            if bearish or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish alignment or ADX < 20 (weakening trend)
            if bullish or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals