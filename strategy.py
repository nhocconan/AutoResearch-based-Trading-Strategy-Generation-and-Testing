#!/usr/bin/env python3

"""
Hypothesis: 6-hour Hull Moving Average crossover with 12-hour RSI filter and volume confirmation.
HMA reduces lag while maintaining smoothness, providing timely trend signals.
12-hour RSI filters out overextended moves, and volume confirms institutional interest.
This combination should work in both bull and bear regimes by focusing on mean-reversion
within the trend context, avoiding excessive whipsaw.
Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma2 = pd.Series(series).ewm(span=half_period, adjust=False).mean()
    wma1 = pd.Series(series).ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma2 - wma1
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
    return hma.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 12h data - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h RSI for filter
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_12h = 100 - (100 / (1 + rs))
    
    # Align 12h RSI to 6h timeframe
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Calculate 6H HMA for entry signals
    hma_fast = calculate_hma(close, 9)
    hma_slow = calculate_hma(close, 21)
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]) or 
            np.isnan(rsi_12h_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: HMA bullish cross, RSI not overbought, volume spike
            if (hma_fast[i] > hma_slow[i] and 
                hma_fast[i-1] <= hma_slow[i-1] and  # Fresh cross
                rsi_12h_aligned[i] < 70 and        # Not overbought
                volume[i] > 1.5 * vol_avg_20[i]):  # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: HMA bearish cross, RSI not oversold, volume spike
            elif (hma_fast[i] < hma_slow[i] and 
                  hma_fast[i-1] >= hma_slow[i-1] and  # Fresh cross
                  rsi_12h_aligned[i] > 30 and        # Not oversold
                  volume[i] > 1.5 * vol_avg_20[i]):  # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: HMA cross in opposite direction
            exit_signal = False
            
            if position == 1:
                # Exit long: HMA bearish cross
                if hma_fast[i] < hma_slow[i] and hma_fast[i-1] >= hma_slow[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: HMA bullish cross
                if hma_fast[i] > hma_slow[i] and hma_fast[i-1] <= hma_slow[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_HMA_Cross_12hRSI_Filter_Volume"
timeframe = "6h"
leverage = 1.0