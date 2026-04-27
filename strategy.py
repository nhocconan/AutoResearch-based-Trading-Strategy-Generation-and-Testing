#!/usr/bin/env python3
"""
6h_ElderRay_ZeroLag_MA_Crossover_v1
Hypothesis: Combines Elder Ray (Bull/Bear Power) with Zero-Lag Moving Average crossovers
on 6h timeframe, filtered by 1w trend direction and volume confirmation.
Elder Ray identifies bullish/bearish power via EMA13, while Zero-Lag MA reduces lag
for timely entries. Weekly trend filter ensures alignment with higher timeframe momentum.
Designed for low trade frequency (15-35 trades/year) to minimize fee drag and work
in both bull and bear markets by capturing momentum shifts with confirmation.
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
    
    # EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Zero-Lag Moving Average (ZLMA) to reduce lag
    # EMA of EMA to cancel lag
    ema1 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema2 = pd.Series(ema1).ewm(span=21, adjust=False, min_periods=21).mean().values
    zlma = 2 * ema1 - ema2
    
    # Signal line (EMA of ZLMA)
    signal_line = pd.Series(zlma).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 1-week trend filter (EMA34 on weekly)
    df_1w = get_htf_data(prices, '1w')
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(zlma[i]) or np.isnan(signal_line[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        zlma_val = zlma[i]
        signal_val = signal_line[i]
        ema34_1w_val = ema34_1w_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long conditions: Bull Power > 0 (bullish pressure), ZLMA crosses above signal line,
            # weekly uptrend, volume confirmation
            if (bull_power[i] > 0 and zlma_val > signal_val and 
                zlma[i-1] <= signal_line[i-1] and  # crossover
                close[i] > ema34_1w_val and vol_conf):
                signals[i] = size
                position = 1
            # Short conditions: Bear Power < 0 (bearish pressure), ZLMA crosses below signal line,
            # weekly downtrend, volume confirmation
            elif (bear_power[i] < 0 and zlma_val < signal_val and 
                  zlma[i-1] >= signal_line[i-1] and  # crossover
                  close[i] < ema34_1w_val and vol_conf):
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: Bear Power < 0 (loss of bullish pressure) or ZLMA crosses below signal
            if bear_power[i] < 0 or (zlma_val < signal_val and zlma[i-1] >= signal_line[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Bull Power > 0 (loss of bearish pressure) or ZLMA crosses above signal
            if bull_power[i] > 0 or (zlma_val > signal_val and zlma[i-1] <= signal_line[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_ZeroLag_MA_Crossover_v1"
timeframe = "6h"
leverage = 1.0