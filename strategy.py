#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with volume confirmation and 1d ADX trend filter.
Long when Alligator is bullish (jaw < teeth < lips) AND volume > 1.3x average AND ADX > 20.
Short when Alligator is bearish (jaw > teeth > lips) AND volume > 1.3x average AND ADX > 20.
Exit when Alligator reverses OR ADX < 15 (range market).
Alligator uses smoothed medians (5,8,13 periods) with future shifts (3,5,8) to avoid look-ahead.
Volume confirmation filters weak breakouts, ADX filter avoids choppy markets.
Target: 75-200 total trades over 4 years (19-50/year). Works in bull markets (captures uptrends) 
and bear markets (captures downtrends) by following the Alligator's alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Alligator calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Alligator on 4h timeframe (using medians with smoothing)
    # Jaw: Blue line - 13-period SMMA of median price, shifted 8 bars
    # Teeth: Red line - 8-period SMMA of median price, shifted 5 bars  
    # Lips: Green line - 5-period SMMA of median price, shifted 3 bars
    median_price = (high_4h + low_4h) / 2
    
    # Smoothed Moving Average (SMMA) - same as RMA/Wilder's MA
    def smma(arr, period):
        sma = pd.Series(arr).rolling(window=period, min_periods=period).mean().values
        # Convert first SMA to EMA-like smoothing for subsequent values
        smoothed = np.full_like(arr, np.nan)
        if len(arr) >= period:
            smoothed[period-1] = sma[period-1]
            for i in range(period, len(arr)):
                if not np.isnan(sma[i]):
                    smoothed[i] = (smoothed[i-1] * (period-1) + sma[i]) / period
                else:
                    smoothed[i] = smoothed[i-1]
        return smoothed
    
    jaw = smma(median_price, 13)  # 13-period SMMA
    teeth = smma(median_price, 8)  # 8-period SMMA
    lips = smma(median_price, 5)   # 5-period SMMA
    
    # Apply Alligator shifts (jaw: +8, teeth: +5, lips: +3)
    # Shift RIGHT to avoid look-ahead (we can only use past data)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set shifted values to NaN for unavailable periods
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    
    # Align 4h Alligator lines to 4h timeframe (no alignment needed - already on 4h)
    jaw_aligned = jaw_shifted
    teeth_aligned = teeth_shifted
    lips_aligned = lips_shifted
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average (20-period) on 4h
    volume_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        
        # Alligator conditions
        bullish = jaw_val < teeth_val < lips_val  # jaw < teeth < lips
        bearish = jaw_val > teeth_val > lips_val  # jaw > teeth > lips
        
        if position == 0:
            # Long: Alligator bullish AND volume > 1.3x avg AND ADX > 20 (trending)
            if bullish and vol > 1.3 * vol_ma and adx_val > 20:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND volume > 1.3x avg AND ADX > 20 (trending)
            elif bearish and vol > 1.3 * vol_ma and adx_val > 20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator turns bearish OR ADX < 15 (range market)
            if not bullish or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator turns bullish OR ADX < 15 (range market)
            if not bearish or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Alligator_Volume_ADX_Filter"
timeframe = "4h"
leverage = 1.0