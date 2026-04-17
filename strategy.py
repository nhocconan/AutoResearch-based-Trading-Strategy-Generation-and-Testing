#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + volume spike + 1d ADX trend filter.
Long when price > Alligator jaws AND volume > 1.5x average AND 1d ADX > 25 (trending).
Short when price < Alligator jaws AND volume > 1.5x average AND 1d ADX > 25.
Exit when price crosses Alligator teeth OR 1d ADX < 20 (range market).
Alligator: Jaw=EMA(13,8), Teeth=EMA(8,5), Lips=EMA(5,3). Uses smoothed median price.
12h timeframe targets 12-37 trades/year. Works in bull markets (captures uptrends via Alligator alignment)
and bear markets (captures downtrends via inverse alignment). Volume confirmation reduces fakeouts.
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
    
    # Calculate median price (typical price) for Alligator
    median_price = (high + low + close) / 3.0
    
    # Get 12h data for Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate median price for 12h
    median_price_12h = (high_12h + low_12h + close_12h) / 3.0
    
    # Calculate Alligator lines on 12h timeframe
    # Jaw: EMA(13,8) - smoothed median price with period 13, shifted 8 bars
    jaw_raw = pd.Series(median_price_12h).ewm(span=13, adjust=False, min_periods=13).mean()
    jaw = jaw_raw.shift(8)  # shift 8 bars forward
    
    # Teeth: EMA(8,5) - smoothed median price with period 8, shifted 5 bars
    teeth_raw = pd.Series(median_price_12h).ewm(span=8, adjust=False, min_periods=8).mean()
    teeth = teeth_raw.shift(5)  # shift 5 bars forward
    
    # Lips: EMA(5,3) - smoothed median price with period 3, shifted 3 bars
    lips_raw = pd.Series(median_price_12h).ewm(span=5, adjust=False, min_periods=5).mean()
    lips = lips_raw.shift(3)  # shift 3 bars forward
    
    # Get 1d data for volume and ADX filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate volume average (20-period) on 1d
    volume_series = pd.Series(volume_1d)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX on 1d timeframe (14-period)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Plus Directional Movement (+DM)
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
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
    
    # Align 12h Alligator lines, 1d volume MA, and 1d ADX to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips.values)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
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
        
        # Alligator alignment: Jaw > Teeth > Lips = uptrend, Jaw < Teeth < Lips = downtrend
        # For simplicity, we use price vs Jaw as primary signal, with teeth/lips for exit
        if position == 0:
            # Long: price > Alligator jaw AND volume > 1.5x avg AND 1d ADX > 25 (trending)
            if price > jaw_val and vol > 1.5 * vol_ma and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: price < Alligator jaw AND volume > 1.5x avg AND 1d ADX > 25 (trending)
            elif price < jaw_val and vol > 1.5 * vol_ma and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Alligator teeth OR 1d ADX < 20 (range market)
            if price < teeth_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Alligator teeth OR 1d ADX < 20 (range market)
            if price > teeth_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_Volume_1dADX_Filter"
timeframe = "12h"
leverage = 1.0