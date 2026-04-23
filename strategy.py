#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA50 Trend and Volume Spike Filter
- Williams Alligator (Jaw=TEETH=13, Teeth=TEETH=8, Lips=TEETH=5) identifies trend presence and direction
- 1d EMA(50) ensures alignment with higher timeframe trend for multi-timeframe confirmation
- Volume > 1.8x 20-period average confirms strong breakout momentum and reduces false signals
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years) to minimize fee drag
- Works in bull markets via Alligator alignment (Lips>Teeth>Jaw) with uptrend, in bear markets via reverse alignment
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
    # Jaw: SMMA(13, 8) - Blue line
    # Teeth: SMMA(8, 5) - Red line  
    # Lips: SMMA(5, 3) - Green line
    def smma(source, period):
        """Smoothed Moving Average"""
        if len(source) < period:
            return np.full_like(source, np.nan)
        result = np.full_like(source, np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + PRICE) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13)  # EMA1d, volume MA, Alligator Jaw
    
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
        # Alligator alignment: Lips > Teeth > Jaw = uptrend (Green > Red > Blue)
        # Alligator alignment: Lips < Teeth < Jaw = downtrend (Green < Red < Blue)
        alligator_long = (lips[i] > teeth[i] and teeth[i] > jaw[i])
        alligator_short = (lips[i] < teeth[i] and teeth[i] < jaw[i])
        
        # Long: Alligator uptrend + price above EMA + volume spike
        # Short: Alligator downtrend + price below EMA + volume spike
        long_signal = (alligator_long and 
                      close[i] > ema_50_1d_aligned[i] and
                      volume[i] > 1.8 * vol_ma[i])
        
        short_signal = (alligator_short and 
                       close[i] < ema_50_1d_aligned[i] and
                       volume[i] > 1.8 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator reverses or price crosses EMA
            exit_signal = False
            
            if position == 1:
                # Exit long: Alligator turns down OR price crosses below EMA
                if not alligator_long or close[i] < ema_50_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Alligator turns up OR price crosses above EMA
                if not alligator_short or close[i] > ema_50_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0