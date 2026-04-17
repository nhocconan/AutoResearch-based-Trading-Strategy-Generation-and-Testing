#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with volume confirmation and 1w ADX trend filter.
Long when price > Alligator Jaw AND Alligator Teeth > Alligator Lips (bullish alignment) 
          AND volume > 1.3x average AND 1w ADX > 20.
Short when price < Alligator Jaw AND Alligator Teeth < Alligator Lips (bearish alignment) 
          AND volume > 1.3x average AND 1w ADX > 20.
Exit when price crosses Alligator Teeth OR 1w ADX < 15 (trend weakens).
Uses 12h for Alligator calculation and 1w for ADX filter to reduce whipsaw.
Target: 50-150 total trades over 4 years (12-37/year). Alligator identifies trends,
volume confirmation filters weak breakouts, weekly ADX ensures strong trend context.
Works in bull markets (captures uptrends) and bear markets (captures downtrends).
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
    
    # Get 12h data for Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Williams Alligator: SMAs of median price (typical price)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA = smoothed moving average (similar to EMA but with different smoothing)
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    
    # Calculate SMMA using EMA with alpha = 1/period (approximation)
    jaw_12h = pd.Series(typical_price_12h).ewm(alpha=1/13, adjust=False).mean().values
    teeth_12h = pd.Series(typical_price_12h).ewm(alpha=1/8, adjust=False).mean().values
    lips_12h = pd.Series(typical_price_12h).ewm(alpha=1/5, adjust=False).mean().values
    
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
    adx_1w = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 12h Alligator to 12h timeframe (no alignment needed)
    jaw_12h_aligned = jaw_12h
    teeth_12h_aligned = teeth_12h
    lips_12h_aligned = lips_12h
    
    # Align 1w ADX to 12h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume average (20-period) on 12h
    volume_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(lips_12h_aligned[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(volume_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        jaw = jaw_12h_aligned[i]
        teeth = teeth_12h_aligned[i]
        lips = lips_12h_aligned[i]
        adx_val = adx_1w_aligned[i]
        vol_ma = volume_ma_12h_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Bullish alignment: price > Jaw AND Teeth > Lips
            bullish = price > jaw and teeth > lips
            # Bearish alignment: price < Jaw AND Teeth < Lips
            bearish = price < jaw and teeth < lips
            
            # Long: bullish alignment AND volume > 1.3x avg AND ADX > 20
            if bullish and vol > 1.3 * vol_ma and adx_val > 20:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment AND volume > 1.3x avg AND ADX > 20
            elif bearish and vol > 1.3 * vol_ma and adx_val > 20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Teeth OR ADX < 15 (trend weakening)
            if price < teeth or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Teeth OR ADX < 15 (trend weakening)
            if price > teeth or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Alligator_Volume_1wADX_Filter"
timeframe = "12h"
leverage = 1.0