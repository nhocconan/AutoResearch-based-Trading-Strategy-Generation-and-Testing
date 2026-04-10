#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume spike + ATR stoploss
# - 4h Donchian channel: upper=20-period high, lower=20-period low
# - Volume confirmation: current volume > 1.8x 20-period average
# - Long: price breaks above Donchian upper AND volume spike
# - Short: price breaks below Donchian lower AND volume spike
# - Exit: ATR-based trailing stop (3*ATR from extreme) OR opposite Donchian break
# - Target: 20-50 trades/year on 4h (80-200 total over 4 years) to avoid fee drag
# - Works in bull/bear: breakouts capture trends, volume filters false signals, ATR stops manage risk

name = "4h_donchian_volume_atr_stop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume MA(20) for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = tr1[0]  # First bar: no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_stop = 0.0   # Trailing stop for longs
    short_stop = 0.0  # Trailing stop for shorts
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition: current volume > 1.8x 20-period average
        volume_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # Break above previous period's high
        breakout_down = close[i] < lowest_low[i-1]   # Break below previous period's low
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up + volume spike
            if breakout_up and volume_spike:
                position = 1
                long_stop = close[i] - 3.0 * atr[i]  # Initial stop
                signals[i] = 0.25
            # Short: Donchian breakout down + volume spike
            elif breakout_down and volume_spike:
                position = -1
                short_stop = close[i] + 3.0 * atr[i]  # Initial stop
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - manage exit
            # Update trailing stop: move up as price makes new highs
            long_stop = max(long_stop, close[i] - 3.0 * atr[i])
            
            # Exit conditions: stop hit OR Donchian breakout down
            stop_hit = close[i] <= long_stop
            breakout_exit = close[i] < lowest_low[i-1]  # Opposite Donchian break
            
            if stop_hit or breakout_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position - manage exit
            # Update trailing stop: move down as price makes new lows
            short_stop = min(short_stop, close[i] + 3.0 * atr[i])
            
            # Exit conditions: stop hit OR Donchian breakout up
            stop_hit = close[i] >= short_stop
            breakout_exit = close[i] > highest_high[i-1]  # Opposite Donchian break
            
            if stop_hit or breakout_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals