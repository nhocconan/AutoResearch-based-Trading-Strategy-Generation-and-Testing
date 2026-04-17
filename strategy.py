#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator strategy with volume confirmation and 1w ADX trend filter.
Long when price > Alligator Jaw (blue line) AND Teeth > Lips (bullish alignment) AND volume > 1.5x average AND weekly ADX > 20.
Short when price < Alligator Jaw AND Teeth < Lips (bearish alignment) AND volume > 1.5x average AND weekly ADX > 20.
Exit when Alligator lines intertwine (jaw between teeth and lips) OR weekly ADX < 15 (weak trend).
Williams Alligator identifies trend presence and direction, volume confirmation reduces false breakouts,
weekly ADX ensures we only trade in established trends across multiple timeframes.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Works in bull markets (captures uptrends via bullish alignment) and bear markets (captures downtrends via bearish alignment).
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
    
    # Get 12h data for Williams Alligator and volume
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw (Blue Line): 13-period SMMA smoothed 8 periods ahead
    # Teeth (Red Line): 8-period SMMA smoothed 5 periods ahead  
    # Lips (Green Line): 5-period SMMA smoothed 3 periods ahead
    
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full_like(data, np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT_DATA) / PERIOD
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Calculate SMMA lines
    jaw = smma(close_12h, 13)  # 13-period SMMA
    teeth = smma(close_12h, 8)  # 8-period SMMA
    lips = smma(close_12h, 5)   # 5-period SMMA
    
    # Apply forward shifts as per Alligator specification
    jaw = np.roll(jaw, 8)   # Jaw shifted 8 bars ahead
    teeth = np.roll(teeth, 5) # Teeth shifted 5 bars ahead
    lips = np.roll(lips, 3)   # Lips shifted 3 bars ahead
    
    # Calculate volume average (20-period) on 12h
    volume_series = pd.Series(volume_12h)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX on 1w timeframe (14-period)
    high_1w_series = pd.Series(high_1w)
    low_1w_series = pd.Series(low_1w)
    close_1w_series = pd.Series(close_1w)
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Plus Directional Movement (+DM)
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smooth / np.where(atr != 0, atr, np.inf))
    minus_di = 100 * (minus_dm_smooth / np.where(atr != 0, atr, np.inf))
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), np.inf)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 12h Alligator lines, volume MA, and 1w ADX to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        vol_ma = volume_ma_aligned[i]
        adx_val = adx_aligned[i]
        vol = volume[i]
        price = close[i]
        
        # Check for Alligator sleeping (intertwined lines) - no trend
        sleeping = (jaw_val > teeth_val and jaw_val < lips_val) or (jaw_val < teeth_val and jaw_val > lips_val)
        
        if position == 0:
            # Long: price > Jaw AND Teeth > Lips (bullish alignment) AND volume > 1.5x avg AND weekly ADX > 20
            if price > jaw_val and teeth_val > lips_val and vol > 1.5 * vol_ma and adx_val > 20 and not sleeping:
                signals[i] = 0.25
                position = 1
            # Short: price < Jaw AND Teeth < Lips (bearish alignment) AND volume > 1.5x avg AND weekly ADX > 20
            elif price < jaw_val and teeth_val < lips_val and vol > 1.5 * vol_ma and adx_val > 20 and not sleeping:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator sleeping OR weekly ADX < 15 (weak trend)
            if sleeping or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator sleeping OR weekly ADX < 15 (weak trend)
            if sleeping or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_Volume_1wADX_Filter"
timeframe = "12h"
leverage = 1.0