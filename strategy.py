#!/usr/bin/env python3
"""
4h_Aroon_MeanReversion_TrendFilter
Hypothesis: Uses Aroon oscillator (25-period) to detect trend exhaustion and mean-reversion opportunities, filtered by 1d EMA50 trend direction and volume confirmation. Enters long when Aroon down crosses above Aroon up (downtrend exhaustion) with price below 1d EMA50, and short when Aroon up crosses above Aroon down (uptrend exhaustion) with price above 1d EMA50. Exits on Aroon crossover reversal or trend reversal. Designed to capture mean-reversion in ranging markets while avoiding trend-following losses in strong trends. Targets 20-40 trades/year via Aroon crossover signals with trend and volume filters.
"""

name = "4h_Aroon_MeanReversion_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_aroon(high, low, period=25):
    """Calculate Aroon Up and Aroon Down"""
    n = len(high)
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(period, n):
        # Periods since highest high
        highest_high_idx = i - np.argmax(high[i-period:i+1])
        aroon_up[i] = ((period - (i - highest_high_idx)) / period) * 100
        
        # Periods since lowest low
        lowest_low_idx = i - np.argmin(low[i-period:i+1])
        aroon_down[i] = ((period - (i - lowest_low_idx)) / period) * 100
    
    return aroon_up, aroon_down

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily EMA50 Trend Filter ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # --- Aroon Oscillator (25-period) ---
    aroon_up, aroon_down = calculate_aroon(high, low, 25)
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h[i]) or np.isnan(aroon_up[i]) or 
            np.isnan(aroon_down[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        # Aroon crossover signals
        aroon_up_cross = (aroon_up[i] > aroon_down[i]) and (aroon_up[i-1] <= aroon_down[i-1])
        aroon_down_cross = (aroon_down[i] > aroon_up[i]) and (aroon_down[i-1] <= aroon_up[i-1])
        
        if position == 0:
            # Long: Aroon down crosses above Aroon up (downtrend exhaustion) + price below 1d EMA50 + volume
            if (aroon_down_cross and 
                close[i] < ema_50_4h[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Aroon up crosses above Aroon down (uptrend exhaustion) + price above 1d EMA50 + volume
            elif (aroon_up_cross and 
                  close[i] > ema_50_4h[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: Aroon up crosses above Aroon down OR trend turns up
                if aroon_up_cross or (close[i] > ema_50_4h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: Aroon down crosses above Aroon up OR trend turns down
                if aroon_down_cross or (close[i] < ema_50_4h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals