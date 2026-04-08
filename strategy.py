#!/usr/bin/env python3
# 4h_donchian20_volatility_breakout_v2
# Hypothesis: Breakouts from Donchian channels with volatility compression (ATR ratio) and volume confirmation.
# Long when price breaks above Donchian(20) high, ATR ratio > 1.5 (expanding volatility), and volume > 1.5x average.
# Short when price breaks below Donchian(20) low, ATR ratio > 1.5, and volume > 1.5x average.
# Exit when price crosses the opposite Donchian band or ATR ratio falls below 0.8 (volatility contraction).
# Uses volatility expansion to capture true breakouts and avoid false signals in ranging markets.
# Target: 20-40 trades/year with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_volatility_breakout_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    lookback = 20
    dc_high = np.full(n, np.nan)
    dc_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        dc_high[i] = np.max(high[i-lookback+1:i+1])
        dc_low[i] = np.min(low[i-lookback+1:i+1])
    
    # ATR for volatility measurement
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]  # First value has no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.full(n, np.nan)
    atr[atr_period-1] = np.mean(tr[0:atr_period])
    for i in range(atr_period, n):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # ATR ratio (current ATR / 50-period ATR average) for volatility regime
    atr_ma_period = 50
    atr_ma = np.full(n, np.nan)
    for i in range(atr_ma_period-1, n):
        atr_ma[i] = np.mean(atr[i-atr_ma_period+1:i+1])
    
    atr_ratio = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(atr[i]) and not np.isnan(atr_ma[i]) and atr_ma[i] > 0:
            atr_ratio[i] = atr[i] / atr_ma[i]
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(lookback, atr_period, atr_ma_period, vol_ma_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or 
            np.isnan(atr_ratio[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below Donchian low OR volatility contraction
            if close[i] < dc_low[i] or atr_ratio[i] < 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above Donchian high OR volatility contraction
            if close[i] > dc_high[i] or atr_ratio[i] < 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above Donchian high, volatility expanding, volume surge
            if (close[i] > dc_high[i] and 
                atr_ratio[i] > 1.5 and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below Donchian low, volatility expanding, volume surge
            elif (close[i] < dc_low[i] and 
                  atr_ratio[i] > 1.5 and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals