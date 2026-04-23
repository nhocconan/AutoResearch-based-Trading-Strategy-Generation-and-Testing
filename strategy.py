#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d EMA34 Trend Filter and Volume Spike
- Uses Williams Alligator (JAW/TEETH/LIPS) for trend direction and entry timing
- 1d EMA34 as higher timeframe trend filter to avoid counter-trend trades
- Volume confirmation (>1.5x 20-period MA) to ensure strong participation
- Discrete position sizing (0.25) to minimize fee churn
- Exits when Alligator lines cross in opposite direction or loses 1d EMA34 trend
- Designed for 4h timeframe to balance trade frequency and noise reduction
- Works in both bull and bear markets via trend filter and volume confirmation
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 4h data
    # JAW: 13-period SMMA, shifted 8 bars forward
    # TEETH: 8-period SMMA, shifted 5 bars forward  
    # LIPS: 5-period SMMA, shifted 3 bars forward
    def smma(data, period):
        """Smoothed Moving Average"""
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT_DATA) / PERIOD
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # First 8 values of jaw_shifted will be from roll - set to nan
    jaw_shifted[:8] = np.nan
    # First 5 values of teeth_shifted
    teeth_shifted[:5] = np.nan
    # First 3 values of lips_shifted
    lips_shifted[:3] = np.nan
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 13)  # need EMA34_1d, vol MA, Alligator
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA34 AND volume spike
            if (lips_shifted[i] > teeth_shifted[i] and 
                teeth_shifted[i] > jaw_shifted[i] and
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA34 AND volume spike
            elif (lips_shifted[i] < teeth_shifted[i] and 
                  teeth_shifted[i] < jaw_shifted[i] and
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator lines cross in opposite direction OR loss of 1d EMA34 trend
            exit_signal = False
            if position == 1:
                # Exit long when Lips < Jaw (bullish alignment broken) OR price < 1d EMA34
                if lips_shifted[i] < jaw_shifted[i] or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when Lips > Jaw (bearish alignment broken) OR price > 1d EMA34
                if lips_shifted[i] > jaw_shifted[i] or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Williams_Alligator_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0