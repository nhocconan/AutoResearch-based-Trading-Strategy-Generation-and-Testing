#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR trailing stop.
# Long when price breaks above 20-period high with volume > 1.5x 20-bar average.
# Short when price breaks below 20-period low with volume > 1.5x 20-bar average.
# Uses ATR(14) trailing stop (2.0x) for risk management.
# Discrete position sizing (0.25) to minimize fee churn.
# Targets 75-200 total trades over 4 years (19-50/year) with strict volume filter.
# Works in both bull/bear markets by requiring volume confirmation to avoid false breakouts.
# Uses 1d HTF only for higher timeframe context (optional trend filter not used to avoid overfitting).

name = "4h_Donchian20_VolumeConfirm_ATRTrail_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels: 20-period high/low
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # ATR for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(lookback, 20)  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price above 20-period high with volume spike
            if curr_close > highest_high[i] and curr_volume_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
            # Short breakout: price below 20-period low with volume spike
            elif curr_close < lowest_low[i] and curr_volume_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_since_entry:
                highest_since_entry = curr_high
            
            # Trailing stop: 2.0 * ATR below highest since entry
            if curr_close < highest_since_entry - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Trailing stop: 2.0 * ATR above lowest since entry
            if curr_close > lowest_since_entry + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals