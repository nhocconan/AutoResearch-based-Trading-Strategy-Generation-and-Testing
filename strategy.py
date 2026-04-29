#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + ATR trailing stop
# Long when price breaks above Donchian(20) high AND volume > 1.5x 20-bar avg volume
# Short when price breaks below Donchian(20) low AND volume > 1.5x 20-bar avg volume
# Exit via ATR trailing stop: 3x ATR from extreme price
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 20-50 trades/year on 4h timeframe.
# Donchian channels provide clear breakout levels, volume confirmation ensures breakout strength,
# ATR trailing stop manages risk and allows trends to run. This combination has worked well on SOL historically.

name = "4h_Donchian20_VolumeConfirm_ATRTrail_v1"
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
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_stop = 0.0
    short_stop = 0.0
    
    start_idx = 20  # Donchian and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        curr_atr = atr[i]
        upper = donch_high[i]
        lower = donch_low[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Update trailing stop: highest high since entry minus 3*ATR
            long_stop = max(long_stop, curr_high - 3.0 * curr_atr)
            # Exit: price hits trailing stop
            if curr_low <= long_stop:
                signals[i] = 0.0
                position = 0
                long_stop = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update trailing stop: lowest low since entry plus 3*ATR
            short_stop = min(short_stop, curr_low + 3.0 * curr_atr)
            # Exit: price hits trailing stop
            if curr_high >= short_stop:
                signals[i] = 0.0
                position = 0
                short_stop = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND volume confirmation
            if curr_high > upper and vol_conf:
                signals[i] = 0.25
                position = 1
                long_stop = curr_high - 3.0 * curr_atr
            # Short when price breaks below Donchian low AND volume confirmation
            elif curr_low < lower and vol_conf:
                signals[i] = -0.25
                position = -1
                short_stop = curr_low + 3.0 * curr_atr
            else:
                signals[i] = 0.0
    
    return signals