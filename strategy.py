#!/usr/bin/env python3
# 6h_1d_alligator_fractal_v1
# Strategy: 6h Williams Alligator + daily fractal reversal with volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Alligator (jaw/teeth/lips) identifies trend absence/presence. Daily fractals
# signal potential reversals. Enter when price crosses Alligator teeth in direction of
# fractal, with volume > 1.5x 20-period average. Works in trends (follow Alligator)
# and reversions (fade at fractal extremes). Low frequency via multi-condition filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_alligator_fractal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 6h Williams Alligator (13,8,5 SMAs with future shift)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # 6x ATR for dynamic thresholds
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily Williams Fractals (requires 2-bar confirmation)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    n_1d = len(high_1d)
    bearish = np.zeros(n_1d, dtype=bool)
    bullish = np.zeros(n_1d, dtype=bool)
    
    for i in range(2, n_1d - 2):
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish[i] = True
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish[i] = True
    
    # Align with 2-bar confirmation delay
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish.astype(float), additional_delay_bars=2)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish.astype(float), additional_delay_bars=2)
    
    # Daily volume average for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or \
           np.isnan(atr[i]) or np.isnan(bearish_aligned[i]) or np.isnan(bullish_aligned[i]) or \
           np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_confirm = vol_1d_aligned[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Alligator conditions: lips > teeth > jaw = bullish, lips < teeth < jaw = bearish
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Price relative to teeth (middle line)
        price_above_teeth = close[i] > teeth[i]
        price_below_teeth = close[i] < teeth[i]
        
        # Fractal signals
        fractal_bear = bearish_aligned[i] > 0.5
        fractal_bull = bullish_aligned[i] > 0.5
        
        # Entry conditions
        # Long: Alligator bullish OR (price crosses above teeth with bullish fractal)
        if ((alligator_bullish and price_above_teeth) or
            (price_above_teeth and fractal_bull and not alligator_bearish)) and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Alligator bearish OR (price crosses below teeth with bearish fractal)
        elif ((alligator_bearish and price_below_teeth) or
              (price_below_teeth and fractal_bear and not alligator_bullish)) and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite fractal or Alligator sleep (all lines intertwined)
        elif position == 1 and (fractal_bear or 
                                (abs(lips[i] - teeth[i]) < 0.001 * close[i] and 
                                 abs(teeth[i] - jaw[i]) < 0.001 * close[i])):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (fractal_bull or 
                                 (abs(lips[i] - teeth[i]) < 0.001 * close[i] and 
                                  abs(teeth[i] - jaw[i]) < 0.001 * close[i])):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals