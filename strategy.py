#!/usr/bin/env python3
"""
4h_1d_camarilla_breakout_v27
Strategy: 4h Camarilla pivot breakout with 1d volume confirmation
Timeframe: 4h
Leverage: 1.0
Hypothesis: Price breaks above Camarilla H4 resistance or below L4 support with volume spike (>1.5x average volume) triggers entry in the direction of breakout. Uses 1d ATR-based volatility filter to avoid low-volatility chop. Designed to capture institutional breakouts that work in both bull (breakouts continue) and bear (breakdowns accelerate) markets. Target: 20-50 trades per year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v27"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_range = prev_high - prev_low
    
    # Camarilla levels (using previous day's close and range)
    # H4 = close + 1.1 * range / 2
    # L4 = close - 1.1 * range / 2
    camarilla_h4 = prev_close + 1.1 * prev_range / 2
    camarilla_l4 = prev_close - 1.1 * prev_range / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Volatility filter: avoid low volatility periods (ATR < 50% of 50-period average)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr > (atr_ma * 0.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(volatility_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Breakout conditions
        breakout_long = price_close > camarilla_h4_aligned[i] and volume_spike[i] and volatility_filter[i]
        breakout_short = price_close < camarilla_l4_aligned[i] and volume_spike[i] and volatility_filter[i]
        
        # Exit when price returns to mid-point (Pivot level)
        pivot_level = (camarilla_h4_aligned[i] + camarilla_l4_aligned[i]) / 2
        exit_long = position == 1 and price_close < pivot_level
        exit_short = position == -1 and price_close > pivot_level
        
        # Trading logic
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Price breaks above Camarilla H4 resistance or below L4 support with volume spike (>1.5x average volume) triggers entry in the direction of breakout. Uses 1d ATR-based volatility filter to avoid low-volatility chop. Designed to capture institutional breakouts that work in both bull (breakouts continue) and bear (breakdowns accelerate) markets. Target: 20-50 trades per year.