#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume spike confirmation.
Long when price > Alligator Jaw (13-period SMMA) and Jaw > Teeth > Lips (bullish alignment) with volume > 2.0x average.
Short when price < Alligator Jaw and Jaw < Teeth < Lips (bearish alignment) with volume > 2.0x average.
Exit on opposite Alligator alignment or trend reversal. Uses 1d timeframe targeting 30-100 total trades over 4 years.
Williams Alligator identifies trend phases, 1w EMA50 filters long-term trend, volume spike confirms breakout strength.
Designed to capture strong momentum moves while avoiding whipsaws in choppy markets across both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if length < 1:
        return np.full_like(source, np.nan, dtype=float)
    result = np.full_like(source, np.nan, dtype=float)
    if len(source) < length:
        return result
    # First value is simple SMA
    result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT_VALUE) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Alligator calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2.0
    
    # Williams Alligator lines (all SMMA)
    jaw_period = 13   # Blue line
    teeth_period = 8  # Red line  
    lips_period = 5   # Green line
    
    jaw = smma(median_price_1d, jaw_period)
    teeth = smma(median_price_1d, teeth_period)
    lips = smma(median_price_1d, lips_period)
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema50_val = ema50_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        # Bullish alignment: Jaw > Teeth > Lips
        bullish_alignment = jaw_val > teeth_val > lips_val
        # Bearish alignment: Jaw < Teeth < Lips
        bearish_alignment = jaw_val < teeth_val < lips_val
        
        if position == 0:
            # Long: price > Jaw AND bullish alignment AND price > 1w EMA50 (uptrend) AND volume spike
            if (price > jaw_val and bullish_alignment and price > ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price < Jaw AND bearish alignment AND price < 1w EMA50 (downtrend) AND volume spike
            elif (price < jaw_val and bearish_alignment and price < ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: bearish alignment OR price < Jaw OR trend reversal
                if (bearish_alignment or price < jaw_val or price < ema50_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: bullish alignment OR price > Jaw OR trend reversal
                if (bullish_alignment or price > jaw_val or price > ema50_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_WilliamsAlligator_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0