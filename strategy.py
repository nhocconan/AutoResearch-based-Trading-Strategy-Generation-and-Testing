#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ATR-based trailing stop
# - Uses 4h Donchian channels for breakout signals (long above 20-period high, short below 20-period low)
# - Confirms with 1d volume > 2.0x 20-period average (strong institutional participation)
# - Exits via ATR trailing stop (3x ATR from extreme) or opposite Donchian touch
# - Position size: 0.25 (25% of capital) to limit drawdown during 2022 crash
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years) to minimize fee drag
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
# - Volume confirmation filters false breakouts; ATR stop manages risk without lookahead

name = "4h_1d_donchian_volume_atrstop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d True Range for ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # 1d ATR(20) for volatility scaling
    atr_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # 1d Volume > 2.0x 20-period average (strict filter to reduce trades)
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * avg_volume_20)
    
    # Align 1d indicators to 4h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: ATR trailing stop or opposite Donchian touch
            if high[i] <= highest_since_entry - (3.0 * atr_1d_aligned[i]):  # ATR trailing stop
                position = 0
                signals[i] = 0.0
            elif low[i] <= donchian_low[i]:  # Touch opposite Donchian band
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: ATR trailing stop or opposite Donchian touch
            if low[i] >= lowest_since_entry + (3.0 * atr_1d_aligned[i]):  # ATR trailing stop
                position = 0
                signals[i] = 0.0
            elif high[i] >= donchian_high[i]:  # Touch opposite Donchian band
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation
            if (high[i] >= donchian_high[i] and  # Break above upper band
                volume_spike_aligned[i]):         # Volume confirmation
                position = 1
                entry_price = high[i]
                atr_stop = atr_1d_aligned[i]
                highest_since_entry = high[i]
                lowest_since_entry = high[i]
                signals[i] = 0.25
            elif (low[i] <= donchian_low[i] and   # Break below lower band
                  volume_spike_aligned[i]):       # Volume confirmation
                position = -1
                entry_price = low[i]
                atr_stop = atr_1d_aligned[i]
                highest_since_entry = low[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals