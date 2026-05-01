#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trailing stop.
# Uses Donchian channel breakouts for entry, volume spike (>2.0x 20-bar MA) for confirmation,
# and ATR(14) trailing stop for risk control. Designed for 4h timeframe to achieve
# 75-200 total trades over 4 years (19-50/year) with discrete sizing (0.30).
# Works in both bull and bear markets via volatility-based breakouts and tight entry conditions.

name = "4h_Donchian20_Breakout_VolumeSpike_ATRStop_v1"
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
    
    # Donchian Channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 2.0)
    
    # ATR(14) for stoploss and position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start after warmup for all indicators
    start_idx = max(lookback, 20, 14) + 1  # 21
    
    for i in range(start_idx, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_ma_20[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Donchian breakout conditions
        breakout_up = curr_high > highest_high[i-1]  # Break above previous period's high
        breakdown_down = curr_low < lowest_low[i-1]  # Break below previous period's low
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up AND volume confirmation
            if breakout_up and vol_confirm:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                atr_stop = entry_price - 2.5 * atr[i]  # 2.5x ATR stoploss
            # Short: Donchian breakdown down AND volume confirmation
            elif breakdown_down and vol_confirm:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                atr_stop = entry_price + 2.5 * atr[i]  # 2.5x ATR stoploss
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update ATR trailing stop (only move up)
            atr_stop = max(atr_stop, curr_close - 2.5 * atr[i])
            # Exit on stoploss hit
            if curr_low <= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Update ATR trailing stop (only move down)
            atr_stop = min(atr_stop, curr_close + 2.5 * atr[i])
            # Exit on stoploss hit
            if curr_high >= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals