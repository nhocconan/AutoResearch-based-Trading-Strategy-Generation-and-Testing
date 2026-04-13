#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and ATR stop.
# Uses Donchian channels (20-period high/low) for breakout signals.
# Daily volume filter ensures breakouts have conviction.
# ATR-based stop loss limits downside risk.
# Target: 100-200 total trades over 4 years (25-50/year) for balanced frequency.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily volume and its 20-period average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR for stop loss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align data to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), donchian_low)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    atr_aligned = align_htf_to_ltf(prices, pd.DataFrame({'tr': tr}), atr)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_ma_20_1d_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 4h volume > 1.5x daily volume MA (adjusted for 4h)
        # 6 4h periods per day, so daily MA/6 = approximate 4h period MA
        volume_4h_approx_ma = volume_ma_20_1d_aligned[i] / 6
        volume_condition = volume[i] > (volume_4h_approx_ma * 1.5)
        
        # Entry conditions: Donchian breakout with volume confirmation
        if position == 0:
            if close[i] > donchian_high_aligned[i] and volume_condition:
                position = 1
                signals[i] = position_size
            elif close[i] < donchian_low_aligned[i] and volume_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price breaks below Donchian low or ATR stop hit
            if close[i] < donchian_low_aligned[i] or close[i] <= (signals[i-1] * position_size * close[i-1] + 2 * atr_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price breaks above Donchian high or ATR stop hit
            if close[i] > donchian_high_aligned[i] or close[i] >= (signals[i-1] * position_size * close[i-1] - 2 * atr_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Donchian_Breakout_Volume_ATR_Stop"
timeframe = "4h"
leverage = 1.0