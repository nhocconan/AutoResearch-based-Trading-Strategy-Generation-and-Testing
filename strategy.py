#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based stoploss.
# Long when price breaks above Donchian upper channel (20-bar high) with volume > 1.5x 20-bar avg.
# Short when price breaks below Donchian lower channel (20-bar low) with volume > 1.5x 20-bar avg.
# Exit when price crosses the Donchian middle (10-bar average of high/low) or ATR stoploss hit.
# Uses volume confirmation to reduce false breakouts and ATR for dynamic risk management.
# Designed to capture trends in both bull and bear markets while minimizing whipsaws.
# Targets 20-50 trades/year on 4h timeframe to avoid fee drag.

name = "4h_Donchian20_VolumeConfirm_ATRStop_v1"
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
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Middle channel: 10-period average of high/low for exit
    high_roll_10 = pd.Series(high).rolling(window=10, min_periods=10).mean().values
    low_roll_10 = pd.Series(low).rolling(window=10, min_periods=10).mean().values
    middle_channel = (high_roll_10 + low_roll_10) / 2
    
    # ATR for stoploss (14-period)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 30  # warmup for Donchian, ATR, and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(middle_channel[i]) or np.isnan(atr[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high_roll[i]
        curr_low = low_roll[i]
        curr_middle = middle_channel[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian with volume confirmation
            if (curr_close > curr_high and curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below lower Donchian with volume confirmation
            elif (curr_close < curr_low and curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: price crosses middle channel OR ATR stoploss hit
            if (curr_close <= curr_middle or 
                curr_close <= entry_price - 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price crosses middle channel OR ATR stoploss hit
            if (curr_close >= curr_middle or 
                curr_close >= entry_price + 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals