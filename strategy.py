#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA50 Trend and Volume Spike Filter
- Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend direction and strength
- 1d EMA(50) ensures alignment with daily trend for multi-timeframe confirmation
- Volume > 2.0x 20-period average confirms breakout momentum while limiting trades
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years) to minimize fee drag
- Works in bull markets via Alligator uptrend (Lips>Teeth>Jaw), in bear markets via Alligator downtrend (Lips<Teeth<Jaw)
- Alligator sleeping condition (all lines intertwined) acts as natural range filter to reduce whipsaw
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
    
    # Get 1d data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    def smma(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        # First value is SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_value) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13)  # EMA1d, volume MA, Alligator jaw
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator signals with trend filter and volume spike
        # Long: Lips > Teeth > Jaw (Alligator eating up) + uptrend + volume spike
        # Short: Lips < Teeth < Jaw (Alligator eating down) + downtrend + volume spike
        # Avoid trading when Alligator is sleeping (all lines intertwined)
        alligator_sleep = (abs(lips[i] - teeth[i]) < (close[i] * 0.001) and 
                          abs(teeth[i] - jaw[i]) < (close[i] * 0.001))
        
        long_signal = (lips[i] > teeth[i] and 
                      teeth[i] > jaw[i] and
                      close[i] > ema_50_1d_aligned[i] and
                      volume[i] > 2.0 * vol_ma[i] and
                      not alligator_sleep)
        
        short_signal = (lips[i] < teeth[i] and 
                       teeth[i] < jaw[i] and
                       close[i] < ema_50_1d_aligned[i] and
                       volume[i] > 2.0 * vol_ma[i] and
                       not alligator_sleep)
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal or Alligator sleeping
            exit_signal = False
            
            if position == 1:
                # Exit long: trend reversal or Alligator starts sleeping
                if (close[i] < ema_50_1d_aligned[i] or 
                    alligator_sleep):
                    exit_signal = True
            elif position == -1:
                # Exit short: trend reversal or Alligator starts sleeping
                if (close[i] > ema_50_1d_aligned[i] or 
                    alligator_sleep):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Williams_Alligator_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0