#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with volume confirmation and ATR-based trailing stop
# Donchian channels identify key support/resistance levels where institutional order flow accumulates.
# Breakouts above/below the 20-period channel with volume spike indicate strong institutional participation.
# ATR trailing stop protects gains while allowing trends to run. Designed for low trade frequency (<30/year)
# to minimize fee drag in both bull and bear markets. Uses discrete position sizing to reduce churn.

name = "12h_Donchian20_Breakout_VolumeSpike_ATRTrail_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate rolling max/min for Donchian channels
    high_roll = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (wait for completed 1d bar)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, high_roll)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, low_roll)
    
    # Calculate ATR(14) for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 30-period average
        vol_ma_30 = np.mean(volume[max(0, i-30):i])
        volume_spike = volume[i] > (2.0 * vol_ma_30)
        
        curr_close = close[i]
        curr_atr = atr[i]
        curr_upper = upper_20_aligned[i]
        curr_lower = lower_20_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: price breaks above 1d Donchian upper channel
                if curr_close > curr_upper:
                    signals[i] = 0.25
                    position = 1
                    highest_since_entry = curr_close
                # Bearish entry: price breaks below 1d Donchian lower channel
                elif curr_close < curr_lower:
                    signals[i] = -0.25
                    position = -1
                    lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest price since entry
            if curr_close > highest_since_entry:
                highest_since_entry = curr_close
            # ATR trailing stop: 2.5 * ATR below highest price since entry
            if curr_close < highest_since_entry - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Exit if price re-enters the Donchian channel (mean reversion signal)
            elif curr_close < curr_upper and curr_close > curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            # ATR trailing stop: 2.5 * ATR above lowest price since entry
            if curr_close > lowest_since_entry + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Exit if price re-enters the Donchian channel (mean reversion signal)
            elif curr_close > curr_lower and curr_close < curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals