#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trailing stop
# Donchian channels provide clear breakout levels that work in both bull and bear markets
# Volume confirmation (>1.5x average) ensures breakout legitimacy and reduces false signals
# ATR trailing stop (3*ATR) manages risk and allows trends to run
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and fee drag

name = "4h_Donchian20_VolumeConfirm_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Need previous bar's channels to avoid look-ahead
    highest_20_prev = np.roll(highest_20, 1)
    lowest_20_prev = np.roll(lowest_20, 1)
    highest_20_prev[0] = np.nan
    lowest_20_prev[0] = np.nan
    
    # Breakout conditions
    breakout_up = close > highest_20_prev
    breakout_down = close < lowest_20_prev
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr3[0] = 0.0  # first bar has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(100, 20, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(highest_20_prev[i]) or 
            np.isnan(lowest_20_prev[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_breakout_up = breakout_up[i]
        curr_breakout_down = breakout_down[i]
        curr_volume_confirm = volume_confirm[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on breakout with volume confirmation
            if curr_volume_confirm:
                # Bullish breakout: price above Donchian high
                if curr_breakout_up:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                # Bearish breakout: price below Donchian low
                elif curr_breakout_down:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_close)
            # ATR trailing stop: exit if price drops 3*ATR from highest since entry
            if curr_close < highest_since_entry - 3.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_close)
            # ATR trailing stop: exit if price rises 3*ATR from lowest since entry
            if curr_close > lowest_since_entry + 3.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals