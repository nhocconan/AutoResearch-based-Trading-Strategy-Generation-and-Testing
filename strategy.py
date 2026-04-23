#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation.
Long when Alligator jaws (13-period SMMA) > teeth (8-period SMMA) > lips (5-period SMMA) AND weekly close > weekly EMA34 AND volume > 2.0x average.
Short when jaws < teeth < lips AND weekly close < weekly EMA34 AND volume > 2.0x average.
Exit when Alligator lines re-interlace (jaws crosses teeth or teeth crosses lips).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-37 trades/year per symbol.
Alligator identifies strong trends, weekly filter avoids counter-trend trades, volume confirms momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if length < 1:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=float)
    # First value is simple average
    if len(source) >= length:
        result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT) / length
    for i in range(length, len(source)):
        if not np.isnan(result[i-1]):
            result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate EMA34 on 1w data for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w indicators to 12h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaws: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2.0
    
    jaws_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply shifts (Alligator specific)
    jaws = np.roll(jaws_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # First values after shift should be NaN (not available yet)
    jaws[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_trend_up = close[i] > ema34_1w_aligned[i]
        weekly_trend_down = close[i] < ema34_1w_aligned[i]
        
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        
        # Alligator alignment conditions
        jaws_above_teeth = jaws[i] > teeth[i]
        teeth_above_lips = teeth[i] > lips[i]
        jaws_below_teeth = jaws[i] < teeth[i]
        teeth_below_lips = teeth[i] < lips[i]
        
        if position == 0:
            # Long: Alligator aligned up (jaws>teeth>lips) AND weekly uptrend AND volume confirmation
            if (jaws_above_teeth and teeth_above_lips and weekly_trend_up and 
                vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down (jaws<teeth<lips) AND weekly downtrend AND volume confirmation
            elif (jaws_below_teeth and teeth_below_lips and weekly_trend_down and 
                  vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator lines re-interlace (trend weakening)
            exit_signal = False
            
            if position == 1:
                # Exit long: jaws crosses below teeth OR teeth crosses below lips
                if not (jaws_above_teeth and teeth_above_lips):
                    exit_signal = True
            else:  # position == -1
                # Exit short: jaws crosses above teeth OR teeth crosses above lips
                if not (jaws_below_teeth and teeth_below_lips):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1wEMA34_Volume"
timeframe = "12h"
leverage = 1.0