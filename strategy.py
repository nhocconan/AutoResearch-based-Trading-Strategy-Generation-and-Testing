#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA(50) trend filter and volume confirmation
# Williams Alligator consists of three SMAs (Jaw=13, Teeth=8, Lips=5) with future shifts
# Long when: Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA(50) AND volume > 1.5x 20-period average
# Short when: Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA(50) AND volume > 1.5x 20-period average
# Uses discrete position sizing (0.25) to minimize fee drag. Alligator works well in trending markets.
# Timeframe: 6h (primary), HTF: 1d for trend filter.

name = "6h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator (SMAs with shifts)
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    def smma(source, period):
        """Smoothed Moving Average"""
        if len(source) < period:
            return np.full_like(source, np.nan, dtype=np.float64)
        result = np.full_like(source, np.nan, dtype=np.float64)
        # First value is SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Apply shifts (Jaw: +8, Teeth: +5, Lips: +3)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Set shifted values to NaN (not available yet)
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 60)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_ema = ema_50_1d_aligned[i]
        curr_lips = lips[i]
        curr_teeth = teeth[i]
        curr_jaw = jaw[i]
        
        # Skip if any Alligator value is NaN
        if np.isnan(curr_lips) or np.isnan(curr_teeth) or np.isnan(curr_jaw):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Alligator loses bullish alignment (Lips <= Teeth or Teeth <= Jaw)
            # 2. Price < 1d EMA(50)
            if (curr_lips <= curr_teeth or 
                curr_teeth <= curr_jaw or
                curr_close < curr_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Alligator loses bearish alignment (Lips >= Teeth or Teeth >= Jaw)
            # 2. Price > 1d EMA(50)
            if (curr_lips >= curr_teeth or 
                curr_teeth >= curr_jaw or
                curr_close > curr_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Bullish alignment: Lips > Teeth > Jaw
            bullish = curr_lips > curr_teeth and curr_teeth > curr_jaw
            # Bearish alignment: Lips < Teeth < Jaw
            bearish = curr_lips < curr_teeth and curr_teeth < curr_jaw
            
            # Long entry: bullish alignment AND price > 1d EMA(50) AND volume spike
            if bullish and curr_close > curr_ema and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish alignment AND price < 1d EMA(50) AND volume spike
            elif bearish and curr_close < curr_ema and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals