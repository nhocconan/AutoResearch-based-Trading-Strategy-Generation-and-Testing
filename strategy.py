#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d EMA50 trend filter and volume spike confirmation.
Long when Alligator jaw < teeth < lips (bullish alignment) AND close > 1d EMA50 AND volume > 2.0x 20-period average.
Short when Alligator jaw > teeth > lips (bearish alignment) AND close < 1d EMA50 AND volume > 2.0x 20-period average.
Exit when Alligator lines cross (jaw-teeth or teeth-lips crossover) or ATR trailing stop (2.0*ATR from extreme).
Williams Alligator identifies trend phases; EMA50 filters higher timeframe trend; volume spike confirms momentum.
Designed for low trade frequency (<30/year) to minimize fee drag while capturing strong trends in both bull and bear markets.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator (13,8,5) smoothed with 8,5,3 periods
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    # Median price for Alligator calculation
    median_price = (high + low) / 2.0
    
    # Jaw (blue line): 13-period SMMA of median, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    jaw = np.roll(jaw, jaw_shift)
    jaw[:jaw_shift] = np.nan
    
    # Teeth (red line): 8-period SMMA of median, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    teeth = np.roll(teeth, teeth_shift)
    teeth[:teeth_shift] = np.nan
    
    # Lips (green line): 5-period SMMA of median, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=lips_period, min_periods=lips_period).mean().values
    lips = np.roll(lips, lips_shift)
    lips[:lips_shift] = np.nan
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(jaw_period + jaw_shift, teeth_period + teeth_shift, lips_period + lips_shift, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema50_val = ema50_1d_aligned[i]
        
        # Alligator alignment conditions
        bullish_alignment = jaw[i] < teeth[i] and teeth[i] < lips[i]
        bearish_alignment = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        if position == 0:
            # Long: Bullish Alligator alignment AND uptrend (close > EMA50) AND volume spike
            if bullish_alignment and close[i] > ema50_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.30
                position = 1
                highest_since_entry = price
            # Short: Bearish Alligator alignment AND downtrend (close < EMA50) AND volume spike
            elif bearish_alignment and close[i] < ema50_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.30
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Alligator lines cross (jaw-teeth or teeth-lips crossover)
            if position == 1 and (jaw[i] >= teeth[i] or teeth[i] >= lips[i]):
                exit_signal = True
            elif position == -1 and (jaw[i] <= teeth[i] or teeth[i] <= lips[i]):
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4H_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_ATRTrailingStop_CrossExit"
timeframe = "4h"
leverage = 1.0