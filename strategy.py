#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + volume confirmation + ATR filter
    # Enter long when price breaks above 20-bar high with volume > 1.5x 20-bar avg volume
    # Enter short when price breaks below 20-bar low with volume > 1.5x 20-bar avg volume
    # Exit on opposite Donchian(10) break or ATR-based stoploss
    # Works in bull (continuation breaks) and bear (reversal breaks at extremes)
    # Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-bar for entry, 10-bar for exit)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # ATR(14) for volatility filter and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(1, n):  # start from 1 to access previous bar
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(high_10[i]) or np.isnan(low_10[i]) or
            np.isnan(atr[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_20[i-1]  # break above previous 20-bar high
        breakout_down = close[i] < low_20[i-1]  # break below previous 20-bar low
        
        # Entry conditions with volume confirmation and ATR filter
        long_entry = breakout_up and volume_confirmed[i] and atr[i] > 0 and position != 1
        short_entry = breakout_down and volume_confirmed[i] and atr[i] > 0 and position != -1
        
        # Exit conditions: opposite Donchian(10) break or ATR stoploss
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long: price breaks below 10-bar low OR 2*ATR stoploss
            exit_long = close[i] < low_10[i-1] or close[i] <= (prices['close'].values[i-1] - 2.0 * atr[i-1]) if i > 0 else False
        elif position == -1:
            # Exit short: price breaks above 10-bar high OR 2*ATR stoploss
            exit_short = close[i] > high_10[i-1] or close[i] >= (prices['close'].values[i-1] + 2.0 * atr[i-1]) if i > 0 else False
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_donchian_breakout_volume_atr_v1"
timeframe = "4h"
leverage = 1.0