#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + Elder Ray with 1d EMA50 trend filter and volume confirmation.
Long when Alligator jaws (blue) < teeth (red) < lips (green) AND Bull Power > 0 AND close > 1d EMA50 AND volume > 1.5x 20-period MA.
Short when Alligator jaws > teeth > lips AND Bear Power < 0 AND close < 1d EMA50 AND volume > 1.5x 20-period MA.
Exit when Alligator reverses (jaws > lips for long, jaws < lips for short) or opposite signal triggers.
Designed for ~12-25 trades/year with trend-following edge in both bull and bear markets.
Alligator identifies trend direction and exhaustion; Elder Ray measures bull/bear power; 1d EMA50 ensures higher timeframe alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator (13,8,5 SMAs with offsets)
    # Jaws (blue): 13-period SMA, offset 8 bars
    # Teeth (red): 8-period SMA, offset 5 bars  
    # Lips (green): 5-period SMA, offset 3 bars
    jaws = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray Index
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13, 8, 5, 20)  # need EMA50, Alligator, Elder Ray, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaws[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1d EMA50 = uptrend, close < 1d EMA50 = downtrend
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter: 12h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Alligator conditions: aligned = jaws < teeth < lips (uptrend), reversed = jaws > teeth > lips (downtrend)
        alligator_aligned = (jaws[i] < teeth[i]) and (teeth[i] < lips[i])
        alligator_reversed = (jaws[i] > teeth[i]) and (teeth[i] > lips[i])
        
        # Elder Ray conditions
        bull_strong = bull_power[i] > 0
        bear_strong = bear_power[i] < 0
        
        # Exit conditions: Alligator reverses or opposite Elder Ray signal
        exit_long = not alligator_aligned or bear_strong
        exit_short = not alligator_reversed or bull_strong
        
        if position == 0:
            # Long: Alligator aligned (jaws<teeth<lips) AND bull power > 0 AND uptrend AND volume confirmation
            if alligator_aligned and bull_strong and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Alligator reversed (jaws>teeth>lips) AND bear power < 0 AND downtrend AND volume confirmation
            elif alligator_reversed and bear_strong and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            if position == 1:
                exit_signal = exit_long
            elif position == -1:
                exit_signal = exit_short
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_ElderRay_1dEMA50_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0