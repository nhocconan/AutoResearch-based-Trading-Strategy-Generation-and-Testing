#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and ATR stoploss
# Long when price breaks above 20-period high + volume > 1.5x average
# Short when price breaks below 20-period low + volume > 1.5x average
# Exit on opposite breakout or when price moves 2*ATR against position
# Uses 4h timeframe to target 75-200 trades over 4 years (19-50/year)
# Volume confirmation reduces false breakouts, works in trending markets

name = "4h_donchian20_vol_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Update stoploss and check exit conditions
        if position == 1:  # long position
            # Exit if price drops 2*ATR below entry or breaks below Donchian low
            if close[i] < entry_price - 2.0 * atr[i] or close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit if price rises 2*ATR above entry or breaks above Donchian high
            if close[i] > entry_price + 2.0 * atr[i] or close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries with volume confirmation
            # Long: price breaks above 20-period high + volume confirmation
            if close[i] > highest_high[i] and volume[i] > volume_threshold[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below 20-period low + volume confirmation
            elif close[i] < lowest_low[i] and volume[i] > volume_threshold[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals