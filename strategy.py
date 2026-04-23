#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d Elder Ray trend filter and volume confirmation.
Long when Alligator jaws (13) < teeth (8) < lips (5) AND 1d Elder Bull Power > 0 AND volume > 1.5x 20-period average.
Short when Alligator jaws > teeth > lips AND 1d Elder Bear Power < 0 AND volume > 1.5x 20-period average.
Exit when Alligator lines re-cross (jaws touches teeth or lips) or ATR stoploss hit (2.5*ATR).
Uses discrete position sizing (0.25) to balance return and risk. Targets 12-37 trades/year per symbol.
Alligator identifies trend emergence, Elder Ray confirms 1d bull/bear power, volume filters weak signals.
Designed for 12h timeframe to reduce trade frequency and fee drag while capturing multi-day swings.
Works in both bull (trend continuation) and bear (mean reversion at extremes) markets.
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
    
    # Calculate Williams Alligator on 12h data (smoothed medians)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).apply(lambda x: np.median(x), raw=True).shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).apply(lambda x: np.median(x), raw=True).shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).apply(lambda x: np.median(x), raw=True).shift(3).values
    
    # Load 1d data for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume average (20-period) on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation (using 12h data)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 13, 8, 5, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        
        if position == 0:
            # Long: Alligator aligned (jaws < teeth < lips) AND Elder Bull Power > 0 AND volume spike
            if (jaw_val < teeth_val and teeth_val < lips_val and 
                bull_val > 0 and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Alligator aligned (jaws > teeth > lips) AND Elder Bear Power < 0 AND volume spike
            elif (jaw_val > teeth_val and teeth_val > lips_val and 
                  bear_val < 0 and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Alligator lines re-cross (jaws touches teeth or lips)
            if position == 1 and (jaw_val >= teeth_val or teeth_val >= lips_val):
                exit_signal = True
            elif position == -1 and (jaw_val <= teeth_val or teeth_val <= lips_val):
                exit_signal = True
            
            # ATR-based stoploss: 2.5 * ATR from entry
            if position == 1 and price < entry_price - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Williams_Alligator_1dElderRay_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0