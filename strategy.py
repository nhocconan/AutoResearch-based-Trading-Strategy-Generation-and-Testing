#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w volume confirmation and ATR trailing stop
# - Uses 20-period 12h Donchian channel breakouts for trend signals
# - Confirms with 1w volume > 2.0x its 20-period average (strong participation)
# - Uses ATR(14) trailing stop: exits when price retraces 3.0x ATR from extreme
# - Position size: 0.25 (25% of capital) to manage drawdown in bear markets
# - Target: 12-30 trades/year on 12h timeframe (50-120 total over 4 years)
# - Donchian breakouts work in trending markets, volume filter reduces false breakouts
# - ATR stop manages risk during volatile periods like 2022 crash
# - Using 1w HTF for volume confirmation reduces noise and increases reliability

name = "12h_1w_donchian_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    volume_1w = df_1w['volume'].values
    
    # 1w volume > 2.0x 20-period average (volume confirmation)
    avg_volume_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_spike_1w = volume_1w > (2.0 * avg_volume_20_1w)
    
    # Align 1w volume spike to 12h
    volume_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w.astype(float))
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume_12h = prices['volume'].values
    
    # 12h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h True Range for ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h[0] = tr_12h[0]
    
    # 12h ATR(14) for volatility and stoploss
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(atr_12h[i]) or
            np.isnan(volume_spike_1w_aligned[i]) or
            atr_12h[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 3.0x ATR from high
            if low[i] <= highest_since_entry - (3.0 * atr_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 3.0x ATR from low
            if high[i] >= lowest_since_entry + (3.0 * atr_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation
            if (high[i] >= donchian_high[i] and    # Break above upper band
                volume_spike_1w_aligned[i]):       # Volume confirmation
                position = 1
                entry_price = high[i]
                highest_since_entry = high[i]
                lowest_since_entry = high[i]  # Initialize for shorts
                signals[i] = 0.25
            elif (low[i] <= donchian_low[i] and    # Break below lower band
                  volume_spike_1w_aligned[i]):     # Volume confirmation
                position = -1
                entry_price = low[i]
                highest_since_entry = low[i]  # Initialize for longs
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals