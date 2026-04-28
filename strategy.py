#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_Filter_VolumeSpike
Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) to identify trend direction on 4h, with RSI(14) for mean-reversion entries during pullbacks, and volume spike confirmation. Designed to capture trend continuation after pullbacks in both bull and bear markets by adapting to market noise.
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
    
    # KAMA parameters
    fast_sc = 0.666  # 2/(2+1)
    slow_sc = 0.0645 # 2/(30+1)
    
    # Calculate Efficiency Ratio (ER) and smoothing constant
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    # Pad arrays to match length
    change = np.concatenate([np.full(9, np.nan), change])
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start at index 9
    for i in range(10, n):
        if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi[14:]])  # Align with close
    
    # Volume spike (>1.5x 20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from KAMA
        trend_up = close[i] > kama[i]
        trend_down = close[i] < kama[i]
        
        # RSI conditions for mean-reversion entries
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # Entry logic:
        # Long: Pullback in uptrend (RSI oversold) with volume spike
        # Short: Pullback in downtrend (RSI overbought) with volume spike
        long_entry = vol_confirm and trend_up and rsi_oversold
        short_entry = vol_confirm and trend_down and rsi_overbought
        
        # Exit logic: Opposite RSI extreme or trend reversal
        long_exit = rsi[i] > 70 or not trend_up
        short_exit = rsi[i] < 30 or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA_Trend_RSI_Filter_VolumeSpike"
timeframe = "4h"
leverage = 1.0