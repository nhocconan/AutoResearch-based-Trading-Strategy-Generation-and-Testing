#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d HTF trend filter and volume confirmation.
Long when price > Alligator Jaw AND Lips > Teeth > Jaw (bullish alignment) AND 1d EMA50 rising AND volume > 1.5x 20-period MA.
Short when price < Alligator Jaw AND Jaw > Teeth > Lips (bearish alignment) AND 1d EMA50 falling AND volume > 1.5x 20-period MA.
Exit when Alligator alignment breaks or price crosses Jaw.
Uses 1d HTF for trend filter to avoid counter-trend trades, Williams Alligator for trend identification + momentum, volume for confirmation.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Williams Alligator identifies trending vs ranging markets, 1d EMA50 filters major trend, volume spike confirms momentum.
Works in bull (trend filters) and bear (volume spikes on breakdowns).
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
    
    # Calculate Williams Alligator (13,8,5 SMAs smoothed)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    # Jaw: 13-period SMA smoothed by 8
    sma_jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    jaw = pd.Series(sma_jaw).rolling(window=8, min_periods=8).mean().values
    
    # Teeth: 8-period SMA smoothed by 5
    sma_teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    teeth = pd.Series(sma_teeth).rolling(window=5, min_periods=5).mean().values
    
    # Lips: 5-period SMA smoothed by 3
    sma_lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().values
    lips = pd.Series(sma_lips).rolling(window=3, min_periods=3).mean().values
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(jaw_period+7, teeth_period+4, lips_period+2, 50, 20)  # Alligator, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 12h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        # Alligator alignment
        bullish_alignment = lips_val > teeth_val and teeth_val > jaw_val
        bearish_alignment = jaw_val > teeth_val and teeth_val > lips_val
        
        if position == 0:
            # Long: Bullish Alligator alignment AND price > Jaw AND EMA50 rising AND volume filter
            if bullish_alignment and price > jaw_val and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment AND price < Jaw AND EMA50 falling AND volume filter
            elif bearish_alignment and price < jaw_val and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Alligator alignment breaks OR price crosses below Jaw
                if not bullish_alignment or price < jaw_val:
                    exit_signal = True
            elif position == -1:
                # Short exit: Alligator alignment breaks OR price crosses above Jaw
                if not bearish_alignment or price > jaw_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1dEMA50_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0