#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using Williams Alligator (Jaw/Teeth/Lips) with 1d trend filter and volume confirmation.
Long when Alligator is bullish (Lips > Teeth > Jaw) AND price > 1d EMA50 (uptrend) AND volume > 1.5x average.
Short when Alligator is bearish (Lips < Teeth < Jaw) AND price < 1d EMA50 (downtrend) AND volume > 1.5x average.
Exit when Alligator becomes neutral (teeth between jaw and lips) OR trend reverses (price crosses 1d EMA50).
Uses 6h timeframe with Alligator's natural smoothing to reduce whipsaw. 1d EMA50 provides higher-timeframe trend filter.
Volume spike ensures breakout conviction. Target: 60-120 trades over 4 years (15-30/year).
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
    
    # Calculate Williams Alligator on 6h (primary timeframe)
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    def smma(src, length):
        if length < 1:
            return np.full_like(src, np.nan)
        result = np.full_like(src, np.nan)
        for i in range(len(src)):
            if i < length - 1:
                continue
            if i == length - 1:
                result[i] = np.nanmean(src[i-length+1:i+1])
            else:
                prev = result[i-1]
                if np.isnan(prev):
                    result[i] = np.nanmean(src[i-length+1:i+1])
                else:
                    result[i] = (prev * (length - 1) + src[i]) / length
        return result
    
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Apply shifts (Jaw: +8, Teeth: +5, Lips: +3)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    # NaN out the rolled values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema50_val = ema50_1d_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        # Alligator conditions
        bullish_alligator = lips_val > teeth_val and teeth_val > jaw_val
        bearish_alligator = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Long: Alligator bullish AND price > 1d EMA50 (uptrend) AND volume spike
            if bullish_alligator and price > ema50_val and vol_current > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND price < 1d EMA50 (downtrend) AND volume spike
            elif bearish_alligator and price < ema50_val and vol_current > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Alligator becomes neutral OR price breaks below 1d EMA50 (trend reversal)
                if not bullish_alligator or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Alligator becomes neutral OR price breaks above 1d EMA50 (trend reversal)
                if not bearish_alligator or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsAlligator_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0