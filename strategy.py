#!/usr/bin/env python3
"""
4h Williams Alligator + Volume Confirmation + ADX Filter
Hypothesis: Williams Alligator identifies trending vs ranging markets. 
When price crosses above/below all three smoothed lines (Jaw, Teeth, Lips) with volume confirmation 
(volume > 1.5x average) and trend strength (ADX > 20), it indicates strong trend continuation.
Williams Alligator works well in both bull and bear markets by filtering out ranging periods.
Target: 25-35 trades/year to minimize fee drain.
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
    
    # Williams Alligator: three smoothed moving averages
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    
    def smma(series, period):
        """Smoothed Moving Average"""
        sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
        smma_vals = np.full_like(series, np.nan, dtype=float)
        if len(series) >= period:
            smma_vals[period-1] = sma[period-1]
            for i in range(period, len(series)):
                smma_vals[i] = (smma_vals[i-1] * (period-1) + series[i]) / period
        return smma_vals
    
    jaw = smma(close, 13)
    teeth = smma(close, 8) 
    lips = smma(close, 5)
    
    # Shift the lines forward (Jaw 8, Teeth 5, Lips 3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Fill NaN from rolling
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # ADX for trend strength (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    di_plus = np.where(tr14 > 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 > 0, 100 * dm_minus14 / tr14, 0)
    
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ema
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        adx_val = adx[i]
        vol_conf = vol_ratio[i] > 1.5
        
        # Bullish alignment: Lips > Teeth > Jaw (price above all)
        bullish_aligned = lips_val > teeth_val and teeth_val > jaw_val
        # Bearish alignment: Lips < Teeth < Jaw (price below all)
        bearish_aligned = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Strong trend with volume confirmation
            # Bullish crossover: price crosses above all three lines
            if adx_val > 20 and bullish_aligned and price > lips_val and vol_conf:
                signals[i] = 0.25
                position = 1
            # Bearish crossover: price crosses below all three lines
            elif adx_val > 20 and bearish_aligned and price < lips_val and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if trend weakens or price crosses back below Teeth
            if adx_val < 15 or price < teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if trend weakens or price crosses back above Teeth
            if adx_val < 15 or price > teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_Volume_ADX"
timeframe = "4h"
leverage = 1.0