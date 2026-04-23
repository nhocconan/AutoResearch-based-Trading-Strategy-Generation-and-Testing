#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1d EMA34 trend filter and volume confirmation.
Long when Alligator is bullish (Lips > Teeth > Jaw) AND price > 1d EMA34 AND volume > 2.0x 20-period MA.
Short when Alligator is bearish (Lips < Teeth < Jaw) AND price < 1d EMA34 AND volume > 2.0x 20-period MA.
Exit when Alligator becomes neutral (Teeth between Jaw and Lips) or volume drops below 1.5x MA.
Uses 1d HTF for trend filter to avoid counter-trend trades, volume spike for momentum confirmation.
Williams Alligator identifies trending vs ranging markets, reducing whipsaws in bear markets like 2022.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
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
    
    # Calculate Williams Alligator on 12h (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    median_12h = (df_12h['high'].values + df_12h['low'].values) / 2
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMMA
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_12h, 13)
    teeth = smma(median_12h, 8)
    lips = smma(median_12h, 5)
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 34, 20)  # Alligator needs 50, EMA34 needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Alligator conditions
        alligator_bullish = lips_val > teeth_val and teeth_val > jaw_val
        alligator_bearish = lips_val < teeth_val and teeth_val < jaw_val
        alligator_neutral = not (alligator_bullish or alligator_bearish)
        
        # Volume filter: 12h volume > 2.0x 20-period MA (strong confirmation)
        vol_filter = volume[i] > 2.0 * vol_ma_val
        vol_exit_filter = volume[i] > 1.5 * vol_ma_val  # Softer exit condition
        
        if position == 0:
            # Long: Alligator bullish AND price > 1d EMA34 AND volume filter
            if alligator_bullish and price > ema_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND price < 1d EMA34 AND volume filter
            elif alligator_bearish and price < ema_val and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Alligator turns neutral OR price < 1d EMA34 OR volume drops
                if alligator_neutral or price < ema_val or not vol_exit_filter:
                    exit_signal = True
            elif position == -1:
                # Short exit: Alligator turns neutral OR price > 1d EMA34 OR volume drops
                if alligator_neutral or price > ema_val or not vol_exit_filter:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0