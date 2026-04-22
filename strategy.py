#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with volume confirmation and ADX trend filter
# Uses 1-day Alligator (Jaw=13, Teeth=8, Lips=5) for trend direction to avoid counter-trend trades
# Entry when price is outside Alligator mouth + volume spike + ADX > 25 (trending market)
# Exit when price re-enters Alligator mouth or ADX < 20
# Target: 20-35 trades/year per symbol, works in bull/bear via trend filter
# Williams Alligator provides clear trend definition with fewer false signals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Williams Alligator and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator components (Smoothed Moving Average)
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
            else:
                result[i] = np.nan
        return result
    
    # Alligator lines: Jaw (13, 8), Teeth (8, 5), Lips (5, 3)
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Shift for future alignment (Alligator uses future values)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Calculate ADX for trend strength
    def calculate_adx(high, low, close, period=14):
        """Calculate ADX (Average Directional Index)"""
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = np.nan
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth TR, +DM, -DM
        def smooth_wilder(arr, period):
            result = np.full_like(arr, np.nan, dtype=float)
            if len(arr) < period:
                return result
            # First value is sum of first 'period' values
            result[period-1] = np.nansum(arr[:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(arr)):
                if not np.isnan(result[i-1]):
                    result[i] = result[i-1] - (result[i-1] / period) + arr[i]
                else:
                    result[i] = np.nan
            return result
        
        atr = smooth_wilder(tr, period)
        plus_di = 100 * smooth_wilder(plus_dm, period) / atr
        minus_di = 100 * smooth_wilder(minus_dm, period) / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = smooth_wilder(dx, period)
        
        return adx
    
    adx = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Volume spike filter (20-period on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 4-hour timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above alligator teeth (bullish alignment) + volume spike + ADX > 25
            if (close[i] > teeth_aligned[i] and lips_aligned[i] > jaw_aligned[i] and 
                vol_spike[i] and adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Price below alligator teeth (bearish alignment) + volume spike + ADX > 25
            elif (close[i] < teeth_aligned[i] and lips_aligned[i] < jaw_aligned[i] and 
                  vol_spike[i] and adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price re-enters alligator mouth or ADX < 20 (trend weakening)
            if position == 1:
                if close[i] < jaw_aligned[i] or adx_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > jaw_aligned[i] or adx_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_Trend_Volume_ADX_Session"
timeframe = "4h"
leverage = 1.0