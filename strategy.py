#!/usr/bin/env python3
"""
4h_RSI_Overbought_Oversold_With_Volume_Confirmation
Hypothesis: Trade RSI extremes with volume confirmation on 4h timeframe. 
Long when RSI < 30 (oversold) with volume spike; short when RSI > 70 (overbought) with volume spike.
Volume confirmation reduces false signals. Works in both bull and bear markets by capturing mean reversion.
Target: 60-120 total trades over 4 years (15-30/year) with position size 0.25.
"""

name = "4h_RSI_Overbought_Oversold_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate RSI with proper min_periods
    def rsi(close_prices, period=14):
        delta = np.diff(close_prices)
        delta = np.concatenate([[np.nan], delta])  # prepend NaN for first element
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        
        # Calculate EMA of up and down moves
        def ema(values, period):
            result = np.full_like(values, np.nan)
            if len(values) >= period:
                multiplier = 2.0 / (period + 1)
                result[period-1] = np.nanmean(values[:period])  # use nanmean to handle initial NaN
                for i in range(period, len(values)):
                    if np.isnan(result[i-1]):
                        result[i] = np.nan
                    else:
                        result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
            return result
        
        up_ema = ema(up, period)
        down_ema = ema(down, period)
        rs = up_ema / down_ema
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    # Calculate volume spike (volume > 1.5x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure RSI and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_vals[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) with volume spike
            if rsi_vals[i] < 30 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) with volume spike
            elif rsi_vals[i] > 70 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (>50) or overbought
            if rsi_vals[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral (<50) or oversold
            if rsi_vals[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals