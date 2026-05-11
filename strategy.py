#!/usr/bin/env python3
"""
4h_RSI_Stochastic_BullBear_RSI50
Hypothesis: Combine RSI(14) with Stochastic Oscillator(14,3,3) and RSI(50) trend filter for mean-reversion entries in both bull and bear markets.
Long when RSI < 30, Stochastic K < 20, and RSI(50) > 50 (bullish bias).
Short when RSI > 70, Stochastic K > 80, and RSI(50) < 50 (bearish bias).
Exit when RSI crosses back above 50 (long) or below 50 (short).
Designed to avoid overtrading by requiring multiple confirmations and using RSI(50) trend filter to avoid counter-trend trades.
Target: 20-50 trades per year on 4h timeframe.
"""

name = "4h_RSI_Stochastic_BullBear_RSI50"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # RSI(14) for entry signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic Oscillator(14,3,3)
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    k_percent_smooth = pd.Series(k_percent).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # RSI(50) for trend filter
    delta_50 = np.diff(close, prepend=close[0])
    gain_50 = np.where(delta_50 > 0, delta_50, 0)
    loss_50 = np.where(delta_50 < 0, -delta_50, 0)
    avg_gain_50 = pd.Series(gain_50).ewm(alpha=1/50, adjust=False, min_periods=50).mean().values
    avg_loss_50 = pd.Series(loss_50).ewm(alpha=1/50, adjust=False, min_periods=50).mean().values
    rs_50 = avg_gain_50 / (avg_loss_50 + 1e-10)
    rsi_50 = 100 - (100 / (1 + rs_50))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi[i]) or 
            np.isnan(k_percent_smooth[i]) or 
            np.isnan(rsi_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 30, Stochastic K < 20, and RSI(50) > 50 (bullish bias)
            if rsi[i] < 30 and k_percent_smooth[i] < 20 and rsi_50[i] > 50:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70, Stochastic K > 80, and RSI(50) < 50 (bearish bias)
            elif rsi[i] > 70 and k_percent_smooth[i] > 80 and rsi_50[i] < 50:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI crosses back above 50
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: RSI crosses back below 50
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals