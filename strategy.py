#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + 1w EMA50 trend filter + volume confirmation
- Williams Alligator (Jaw=13, Teeth=8, Lips=5) from 6h defines market structure
- Only trade when all three lines are aligned (bullish: Lips>Teeth>Jaw, bearish: Lips<Teeth<Jaw)
- 1w EMA50 defines higher timeframe trend: only trade Alligator signals in weekly trend direction
- Volume confirmation (> 1.5x 20-period average) filters false signals
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading with the 1w trend and Alligator alignment
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
    
    # Calculate 6h Williams Alligator
    # Jaw: 13-period SMMA (smoothed moving average) of median price
    # Teeth: 8-period SMMA of median price  
    # Lips: 5-period SMMA of median price
    median_price = (high + low) / 2
    
    def smma(source, period):
        """Smoothed Moving Average"""
        if len(source) < period:
            return np.full_like(source, np.nan)
        result = np.full_like(source, np.nan)
        # First value is SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13)  # for EMA50, volume MA, and Alligator
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish Alligator alignment: Lips > Teeth > Jaw
            bullish_alligator = (lips[i] > teeth[i] and teeth[i] > jaw[i])
            # Bearish Alligator alignment: Lips < Teeth < Jaw
            bearish_alligator = (lips[i] < teeth[i] and teeth[i] < jaw[i])
            
            # Long conditions: bullish Alligator + 1w uptrend + volume
            long_signal = (bullish_alligator and 
                          close[i] > ema_50_1w_aligned[i] and
                          volume[i] > 1.5 * vol_ma[i])
            
            # Short conditions: bearish Alligator + 1w downtrend + volume
            short_signal = (bearish_alligator and 
                           close[i] < ema_50_1w_aligned[i] and
                           volume[i] > 1.5 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator misalignment or trend reversal
            exit_signal = False
            
            # Check for Alligator misalignment (lines intertwined)
            bullish_alligator = (lips[i] > teeth[i] and teeth[i] > jaw[i])
            bearish_alligator = (lips[i] < teeth[i] and teeth[i] < jaw[i])
            alligator_misaligned = not (bullish_alligator or bearish_alligator)
            
            if position == 1:
                # Exit long: Alligator misaligned or 1w trend turns bearish
                if (alligator_misaligned or 
                    close[i] < ema_50_1w_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: Alligator misaligned or 1w trend turns bullish
                if (alligator_misaligned or 
                    close[i] > ema_50_1w_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsAlligator_1wEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0