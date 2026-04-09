#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and ATR trailing stop
# - Uses 12h Donchian channel breakout (20-period) for entry signals
# - Confirms with 1d volume > 2.0x its 20-period average (strong participation)
# - Uses ATR(14) trailing stop: exits when price retraces 2.5x ATR from extreme
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Donchian breakouts work in trending markets; volume filter reduces false signals
# - ATR stop manages risk in volatile markets, especially during 2022 crash
# - Primary timeframe: 12h, HTF: 1d for volume confirmation

name = "12h_1d_donchian_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d indicators (volume confirmation)
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * avg_volume_20)
    
    # Align 1d volume spike to 12h
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Pre-compute 12h indicators (Donchian and ATR)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # 12h Donchian channel (20-period)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # 12h True Range for ATR
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h[0] = tr_12h[0]
    
    # 12h ATR(14) for volatility and stoploss
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_12h[i]) or np.isnan(volume_spike_1d_aligned[i]) or
            atr_12h[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high_12h[i] > highest_since_entry:
                highest_since_entry = high_12h[i]
            
            # Exit conditions: price retraces 2.5x ATR from high
            if low_12h[i] <= highest_since_entry - (2.5 * atr_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low_12h[i] < lowest_since_entry:
                lowest_since_entry = low_12h[i]
            
            # Exit conditions: price retraces 2.5x ATR from low
            if high_12h[i] >= lowest_since_entry + (2.5 * atr_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation
            if (high_12h[i] >= donchian_high[i] and    # Break above upper band
                volume_spike_1d_aligned[i]):           # Volume confirmation
                position = 1
                highest_since_entry = high_12h[i]
                lowest_since_entry = high_12h[i]  # Initialize for shorts
                signals[i] = 0.25
            elif (low_12h[i] <= donchian_low[i] and    # Break below lower band
                  volume_spike_1d_aligned[i]):         # Volume confirmation
                position = -1
                highest_since_entry = low_12h[i]  # Initialize for longs
                lowest_since_entry = low_12h[i]
                signals[i] = -0.25
    
    return signals