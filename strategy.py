#!/usr/bin/env python3
"""
4h_KAMA_Trend_Filter_Volume_Spike_Entry
Hypothesis: 4h strategy using Kaufman Adaptive Moving Average (KAMA) for trend direction (fast/slow adaptation) combined with volume spike (>2.0x 20-bar mean) for entry confirmation. Enters long when price > KAMA and volume spike, short when price < KAMA and volume spike. Uses discrete position sizing (0.25) to minimize fee drag. Designed for 20-40 trades/year per symbol, effective in both bull (trend following) and bear (counter-trend reversals on volume spikes) markets.
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
    
    # Calculate KAMA(10, 2, 30) on close prices
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Proper KAMA calculation
    dir = np.abs(np.diff(close, 10))  # direction over 10 periods
    vol = np.sum(np.abs(np.diff(close, 1)), axis=1) if False else None  # placeholder
    
    # Manual KAMA implementation for efficiency
    close_series = pd.Series(close)
    diff = close_series.diff(1).abs()
    volatility = diff.rolling(window=10, min_periods=1).sum()
    direction = np.abs(close_series - close_series.shift(10))
    er = direction / volatility.replace(0, np.nan)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    sc = sc.fillna(0).values
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: current volume > 2.0x 20-bar mean volume
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA and volume mean
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama[i]) or np.isnan(vol_mean_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price > KAMA with volume confirmation
            # Short: price < KAMA with volume confirmation
            long_signal = (close[i] > kama[i]) and vol_confirm[i]
            short_signal = (close[i] < kama[i]) and vol_confirm[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below KAMA (trend reversal)
            exit_signal = close[i] < kama[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above KAMA (trend reversal)
            exit_signal = close[i] > kama[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Trend_Filter_Volume_Spike_Entry"
timeframe = "4h"
leverage = 1.0