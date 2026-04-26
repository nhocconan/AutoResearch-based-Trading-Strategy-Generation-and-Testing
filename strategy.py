#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_ATRStop_v1
Hypothesis: On 4h timeframe, enter long when price breaks above Donchian(20) upper band AND volume > 2.0x 20-period average volume. Enter short when price breaks below Donchian(20) lower band AND volume > 2.0x 20-period average volume. Exit via ATR-based trailing stop (3*ATR) or opposite Donchian breakout. Uses discrete sizing (0.0, ±0.25) to limit fee drag. Donchian breakouts capture strong momentum moves, volume confirmation ensures participation, and ATR stop manages risk. Designed to generate ~20-50 trades per year on BTC/ETH/SOL with Sharpe > 0 in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: need Donchian(20), ATR(14), volume MA(20) warmup
    start_idx = max(20, 14, 20)  # 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        if position == 0:
            # Long: breakout above upper band + volume spike
            if breakout_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                highest_since_long = close[i]
            # Short: breakout below lower band + volume spike
            elif breakout_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                lowest_since_short = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Update highest close since entry
            if close[i] > highest_since_long:
                highest_since_long = close[i]
            # Exit conditions: ATR stoploss OR opposite breakout
            if close[i] < highest_since_long - 3.0 * atr[i] or breakout_down:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Update lowest close since entry
            if close[i] < lowest_since_short:
                lowest_since_short = close[i]
            # Exit conditions: ATR stoploss OR opposite breakout
            if close[i] > lowest_since_short + 3.0 * atr[i] or breakout_up:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0