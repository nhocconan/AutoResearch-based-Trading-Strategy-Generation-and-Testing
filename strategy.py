#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Williams Alligator (Jaw/Teeth/Lips) for trend direction,
# combined with 1d volume spike and ADX trend strength filter.
# Long when Alligator is bullish (Lips > Teeth > Jaw), volume > 2x 20-period average, and ADX > 25.
# Short when Alligator is bearish (Lips < Teeth < Jaw), volume > 2x 20-period average, and ADX > 25.
# Exit when Alligator direction becomes neutral or volume drops below average.
# Uses discrete position size 0.25. Alligator provides smoothed trend, volume confirms momentum,
# ADX ensures trending regime to avoid whipsaws. Target: 30-100 total trades over 4 years (7-25/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once before loop for Alligator and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: Williams Alligator (SMMA-based) ===
    # Smoothed Moving Average (SMMA) - similar to Wilder's smoothing
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Alligator periods: Jaw=13, Teeth=8, Lips=5 (all shifted forward)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    jaw = smma(close_1w, jaw_period)
    teeth = smma(close_1w, teeth_period)
    lips = smma(close_1w, lips_period)
    
    # Shift Alligator lines forward by their respective periods (as per Williams)
    jaw = np.roll(jaw, -jaw_period)
    teeth = np.roll(teeth, -teeth_period)
    lips = np.roll(lips, -lips_period)
    
    # Alligator direction: 1=bullish (Lips > Teeth > Jaw), -1=bearish (Lips < Teeth < Jaw), 0=neutral/entanglement
    alligator_dir = np.zeros_like(close_1w)
    bullish = (lips > teeth) & (teeth > jaw)
    bearish = (lips < teeth) & (teeth < jaw)
    alligator_dir[bullish] = 1
    alligator_dir[bearish] = -1
    
    # === 1w Indicators: ADX (Trend Strength) ===
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = SMMA)
    atr = smma(tr, 14)
    plus_dm_sm = smma(plus_dm, 14)
    minus_dm_sm = smma(minus_dm, 14)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_sm / atr
    minus_di = 100 * minus_dm_sm / atr
    
    # DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = smma(dx, 14)
    
    # Get 1d data once before loop for volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1d timeframe
    alligator_dir_aligned = align_htf_to_ltf(prices, df_1w, alligator_dir)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(alligator_dir_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        alligator_dir_val = alligator_dir_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Alligator turns neutral/bearish or ADX weakens
            if alligator_dir_val <= 0 or adx_val < 20:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Alligator turns neutral/bullish or ADX weakens
            if alligator_dir_val >= 0 or adx_val < 20:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Trend filter: Alligator must show clear direction
            trend_filter = alligator_dir_val != 0
            
            # Volume filter: volume > 2x 20-period average (1d)
            vol_filter = vol > 2.0 * vol_ma_val
            
            # Trend strength filter: ADX > 25 (strong trend)
            adx_filter = adx_val > 25
            
            # LONG: Alligator bullish, volume spike, strong trend
            if (alligator_dir_val > 0) and vol_filter and adx_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Alligator bearish, volume spike, strong trend
            elif (alligator_dir_val < 0) and vol_filter and adx_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_1wAlligator_ADX_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0