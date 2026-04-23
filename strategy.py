#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA50 Trend and Volume Spike Filter
- Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trending vs ranging markets
- 1d EMA(50) ensures alignment with higher timeframe trend for multi-timeframe confirmation
- Volume > 1.8x 20-period average confirms strong breakout momentum and reduces false signals
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years) to minimize fee drag
- Works in bull markets via Alligator alignment with trend, in bear markets via mean reversion at extreme levels
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
    
    # Get 1d data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA of median price, shifted 8 bars forward
    # Teeth: 8-period SMMA of median price, shifted 5 bars forward  
    # Lips: 5-period SMMA of median price, shifted 3 bars forward
    median_price = (high + low) / 2
    
    def smma(source, period):
        """Smoothed Moving Average"""
        if len(source) < period:
            return np.full_like(source, np.nan)
        result = np.full_like(source, np.nan)
        sma = np.nansum(source[:period]) / period
        result[period-1] = sma
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13)  # EMA1d, volume MA, Alligator
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator signals with trend filter and volume spike
        # Alligator is sleeping (ranging) when lines are intertwined
        # Alligator is awake (trending) when lines are separated and aligned
        # Long: Lips > Teeth > Jaw (bullish alignment) + uptrend + volume spike
        # Short: Lips < Teeth < Jaw (bearish alignment) + downtrend + volume spike
        bullish_alignment = (lips[i] > teeth[i] and teeth[i] > jaw[i])
        bearish_alignment = (lips[i] < teeth[i] and teeth[i] < jaw[i])
        
        long_signal = bullish_alignment and (close[i] > ema_50_1d_aligned[i]) and (volume[i] > 1.8 * vol_ma[i])
        short_signal = bearish_alignment and (close[i] < ema_50_1d_aligned[i]) and (volume[i] > 1.8 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator starts to sleep (lines intertwine) or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: bearish alignment or trend reversal
                if bearish_alignment or (close[i] < ema_50_1d_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: bullish alignment or trend reversal
                if bullish_alignment or (close[i] > ema_50_1d_aligned[i]):
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