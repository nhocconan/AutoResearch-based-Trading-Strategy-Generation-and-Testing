#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and ATR trailing stop
# - Uses 1d Donchian channel (20-period high/low) for breakout signals
# - Confirms with 1w volume > 1.5x its 10-period average (strong weekly participation)
# - Uses ATR(14) trailing stop: exits when price retraces 3.0x ATR from extreme
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 15-30 trades/year on 1d timeframe (60-120 total over 4 years)
# - Donchian breakouts capture strong trends; volume filter reduces false signals
# - Weekly timeframe provides structural context for daily breakouts
# - Works in both bull (breakouts continuation) and bear (breakdowns continuation) markets

name = "1d_1w_donchian_volume_atr_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d True Range for ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    
    # 1d ATR(14) for volatility and stoploss
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1w volume > 1.5x 10-period average (volume confirmation)
    volume_1w = df_1w['volume'].values
    avg_volume_10 = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    volume_spike_1w = volume_1w > (1.5 * avg_volume_10)
    
    # Align 1w indicators to 1d
    volume_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(atr[i]) or
            np.isnan(volume_spike_1w_aligned[i]) or
            atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 3.0x ATR from high
            if low[i] <= highest_since_entry - (3.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 3.0x ATR from low
            if high[i] >= lowest_since_entry + (3.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation
            if (high[i] >= donchian_high[i] and    # Break above Donchian high
                volume_spike_1w_aligned[i]):       # Weekly volume confirmation
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = high[i]
                signals[i] = 0.25
            elif (low[i] <= donchian_low[i] and    # Break below Donchian low
                  volume_spike_1w_aligned[i]):     # Weekly volume confirmation
                position = -1
                highest_since_entry = low[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals