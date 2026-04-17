#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with 1w Williams Alligator (Jaw/Teeth/Lips) + volume confirmation + ADX regime filter.
Long when Lips cross above Teeth with volume confirmation and ADX > 25 (trending).
Short when Lips cross below Teeth with volume confirmation and ADX > 25 (trending).
Exit when Lips cross back over Teeth or ADX < 20 (range regime).
Uses 1w timeframe for structure (reduces noise) and 1d for entry timing and volume confirmation.
Designed to capture strong trends while avoiding whipsaws in ranging markets.
Williams Alligator provides smoothed trend identification effective in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator calculation (based on 1d candles)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator on 1d timeframe
    # Jaw: 13-period SMMA smoothed 8 bars ahead
    # Teeth: 8-period SMMA smoothed 5 bars ahead  
    # Lips: 5-period SMMA smoothed 3 bars ahead
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CLOSE) / PERIOD
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_raw = smma(close_1d, 13)
    teeth_raw = smma(close_1d, 8)
    lips_raw = smma(close_1d, 5)
    
    # Apply smoothing offsets (Jaw +8, Teeth +5, Lips +3)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Calculate 1d volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX for regime filter (trending when ADX > 25)
    def calculate_adx(high, low, close, period=14):
        """Calculate ADX (Average Directional Index)"""
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original arrays
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed TR, +DM, -DM
        def smma_arr(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan, dtype=float)
            result = np.full_like(arr, np.nan, dtype=float)
            # First value is simple average of first 'period' elements (skip nan)
            valid_start = period
            while valid_start < len(arr) and np.isnan(arr[valid_start-1]):
                valid_start += 1
            if valid_start >= len(arr):
                return result
            # Find first valid index for SMA
            first_valid = 0
            while first_valid < len(arr) and np.isnan(arr[first_valid]):
                first_valid += 1
            if first_valid + period > len(arr):
                return result
            result[first_valid + period - 1] = np.nanmean(arr[first_valid:first_valid+period])
            # Subsequent values
            for i in range(first_valid + period, len(arr)):
                if np.isnan(result[i-1]):
                    result[i] = np.nan
                else:
                    result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result
        
        atr = smma_arr(tr, period)
        plus_dm_sm = smma_arr(plus_dm, period)
        minus_dm_sm = smma_arr(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_sm / atr
        minus_di = 100 * minus_dm_sm / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = smma_arr(dx, period)
        return adx
    
    adx = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d indicators to 1d timeframe (same timeframe, so no alignment needed for indicators)
    # But we still use align_htf_to_ltf for proper handling of HTF data
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for Alligator smoothing and ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Regime filter: ADX > 25 for trending market
        is_trending = adx_aligned[i] > 25
        
        if position == 0:
            # Long: Lips cross above Teeth with volume and trending regime
            if (lips_aligned[i] > teeth_aligned[i] and 
                lips_aligned[i-1] <= teeth_aligned[i-1] and  # crossed above
                volume_confirmed and 
                is_trending):
                signals[i] = 0.25
                position = 1
            # Short: Lips cross below Teeth with volume and trending regime
            elif (lips_aligned[i] < teeth_aligned[i] and 
                  lips_aligned[i-1] >= teeth_aligned[i-1] and  # crossed below
                  volume_confirmed and 
                  is_trending):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Lips cross back below Teeth OR ADX < 20 (range regime)
            if (lips_aligned[i] < teeth_aligned[i] and 
                lips_aligned[i-1] >= teeth_aligned[i-1]) or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Lips cross back above Teeth OR ADX < 20 (range regime)
            if (lips_aligned[i] > teeth_aligned[i] and 
                lips_aligned[i-1] <= teeth_aligned[i-1]) or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1wWilliamsAlligator_Volume_ADX_Regime"
timeframe = "1d"
leverage = 1.0