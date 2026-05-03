#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR trailing stop.
# Long when: price breaks above 20-period Donchian high AND volume > 1.5x 24-bar average
# Short when: price breaks below 20-period Donchian low AND volume > 1.5x 24-bar average
# Exit via ATR(24) trailing stop: long exit when price < highest_high_since_entry - 2.0 * ATR
#                      short exit when price > lowest_low_since_entry + 2.0 * ATR
# Uses proven Donchian breakout structure (works in bull/bear), volume confirmation for validity,
# ATR stop for risk management. Discrete sizing 0.25 targets ~100 trades/year.
# This avoids Camarilla saturation while capturing similar price channel edge.

name = "4h_Donchian20_VolumeSpike_ATRStop_v1"
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
    
    # 4h Donchian(20) channels (based on completed 4h bars)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:  # Need 20 for lookback + 1 current
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channels: 20-period high/low of completed 4h bars
    donchian_high = np.zeros(len(close_4h))
    donchian_low = np.zeros(len(close_4h))
    
    for i in range(len(close_4h)):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            # Use last 20 completed 4h bars (indices i-20 to i-1)
            donchian_high[i] = np.max(high_4h[i-20:i])
            donchian_low[i] = np.min(low_4h[i-20:i])
    
    # Align 4h Donchian levels to 4h timeframe (completed 4h bar only)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume confirmation (1.5x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    # 4h ATR(24) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=24, adjust=False, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for Donchian, volume, ATR calculations)
    start_idx = 24 + 20 + 5  # ATR(24) + Donchian(20) warmup + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above 4h Donchian high with volume spike
            if close[i] > donchian_high_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry = high[i]
            # Short entry: price breaks below 4h Donchian low with volume spike
            elif close[i] < donchian_low_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = low[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # ATR trailing stop: exit when price < highest_high_since_entry - 2.0 * ATR
            if close[i] < highest_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit when price > lowest_low_since_entry + 2.0 * ATR
            if close[i] > lowest_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals