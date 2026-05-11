#!/usr/bin/env python3
"""
4h_SuperTrend_RSI_Filter
Hypothesis: Uses SuperTrend(10,3) for trend direction and RSI(14) for overbought/oversold conditions. 
Enters long in uptrend when RSI < 40, exits when RSI > 60. Enters short in downtrend when RSI > 60, 
exits when RSI < 40. Includes volume confirmation (volume > 1.5x 20-period average) to filter false signals.
Designed for low trade frequency with clear signals in both bull and bear markets by following the higher 
timeframe trend and avoiding choppy markets.
"""

name = "4h_SuperTrend_RSI_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # SuperTrend calculation (ATR=10, multiplier=3)
    atr_period = 10
    multiplier = 3
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR
    atr = np.zeros_like(close)
    atr[atr_period] = np.mean(tr[:atr_period+1])
    for i in range(atr_period+1, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Basic Upper and Lower Bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # SuperTrend
    super_trend = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    super_trend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > super_trend[i-1]:
            direction[i] = 1
        elif close[i] < super_trend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            super_trend[i] = max(lower_band[i], super_trend[i-1])
        else:
            super_trend[i] = min(upper_band[i], super_trend[i-1])
    
    # RSI calculation
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    for i in range(rsi_period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)
    rsi = np.where(avg_gain == 0, 0, rsi)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(atr_period, rsi_period, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(super_trend[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: uptrend + RSI oversold + volume confirmation
            if (direction[i] == 1 and 
                rsi[i] < 40 and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: downtrend + RSI overbought + volume confirmation
            elif (direction[i] == -1 and 
                  rsi[i] > 60 and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: RSI overbought OR trend reversal
                if (rsi[i] > 60) or (direction[i] == -1):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI oversold OR trend reversal
                if (rsi[i] < 40) or (direction[i] == 1):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals