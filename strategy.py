#!/usr/bin/env python3
name = "4h_RSI_MeanReversion_Volume_Filter"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14) for mean reversion
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to alpha=1/14)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: RSI oversold with volume confirmation
            if rsi[i] < 30 and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought with volume confirmation
            elif rsi[i] > 70 and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI returns to neutral or overbought
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI returns to neutral or oversold
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: RSI mean reversion with volume filter on 4h timeframe
# - RSI < 30 indicates oversold conditions (long opportunity)
# - RSI > 70 indicates overbought conditions (short opportunity)
# - Volume confirmation (1.5x average) reduces false signals
# - Exit when RSI returns to neutral (50) to avoid giving back profits
# - Works in both bull and bear markets as it captures mean reversion
# - Volume filter ensures participation only during active market periods
# - Position size 0.25 targets ~30-50 trades/year to minimize fee drag
# - Simple, robust strategy with clear entry/exit rules
# - Aims for low trade frequency with high win rate through confluence of RSI extremes and volume
# - Avoids overtrading by requiring both RSI extreme and volume confirmation simultaneously