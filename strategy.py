#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
Long when Alligator jaws (13-period SMMA) crosses above teeth (8-period SMMA) and price > 1d EMA34 (uptrend) with volume > 1.5x average.
Short when jaws cross below teeth and price < 1d EMA34 (downtrend) with volume > 1.5x average.
Exit on opposite crossover or trend reversal. Uses 12h timeframe targeting 50-150 total trades over 4 years.
Williams Alligator identifies trend initiation and continuation, EMA34 filters higher-timeframe trend, volume confirms breakout strength.
Designed to capture strong momentum moves while avoiding whipsaws in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (SMMA) - also called Wilder's MA"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    result = np.full_like(arr, np.nan, dtype=float)
    # First value is SMA
    result[period-1] = np.mean(arr[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_PRICE) / period
    for i in range(period, len(arr)):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Williams Alligator - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Williams Alligator components on 12h
    jaws_12h = smma(close_12h, 13)  # 13-period SMMA
    teeth_12h = smma(close_12h, 8)   # 8-period SMMA
    lips_12h = smma(close_12h, 5)    # 5-period SMMA (not used in signals but part of Alligator)
    
    # Align 12h Alligator to 12h timeframe (no alignment needed as we're using 12h data on 12h timeframe)
    # But we need to align to the primary timeframe which is 12h, so we can use directly
    jaws_12h_aligned = jaws_12h
    teeth_12h_aligned = teeth_12h
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaws_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaws_val = jaws_12h_aligned[i]
        teeth_val = teeth_12h_aligned[i]
        ema34_val = ema34_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: jaws cross above teeth AND price > 1d EMA34 (uptrend) AND volume spike
            if (jaws_val > teeth_val and jaws_12h_aligned[i-1] <= teeth_12h_aligned[i-1] and 
                price > ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: jaws cross below teeth AND price < 1d EMA34 (downtrend) AND volume spike
            elif (jaws_val < teeth_val and jaws_12h_aligned[i-1] >= teeth_12h_aligned[i-1] and 
                  price < ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: jaws cross below teeth OR trend reversal
                if (jaws_val < teeth_val and jaws_12h_aligned[i-1] >= teeth_12h_aligned[i-1]) or price < ema34_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: jaws cross above teeth OR trend reversal
                if (jaws_val > teeth_val and jaws_12h_aligned[i-1] <= teeth_12h_aligned[i-1]) or price > ema34_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Williams_Alligator_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0