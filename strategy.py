#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w volume spike and ATR trailing stop
# - Uses 12h Donchian channel (20-period high/low) for breakout signals
# - Confirms with 1w volume > 2.5x its 20-period average (strong participation)
# - Uses ATR(14) trailing stop: exits when price retraces 2.0x ATR from extreme
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
# - Donchian channels adapt to volatility and provide objective breakout levels
# - Volume filter reduces false breakouts, ATR stop manages risk

name = "12h_1w_donchian_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # 1w True Range for ATR
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w[0] = tr_1w[0]
    
    # 1w ATR(14) for volatility and stoploss
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # 1w Volume > 2.5x 20-period average (stricter for fewer trades)
    avg_volume_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_spike_1w = volume_1w > (2.5 * avg_volume_20)
    
    # Align 1w indicators to 12h
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    volume_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w.astype(float))
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_1w_aligned[i]) or np.isnan(volume_spike_1w_aligned[i]) or
            atr_1w_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high
            if low[i] <= highest_since_entry - (2.0 * atr_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low
            if high[i] >= lowest_since_entry + (2.0 * atr_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Calculate 12h Donchian channels using 20-period lookback
            # 20 periods of 12h = 10 periods of 1w (since 1w = 14 * 12h)
            lookback_periods = 10  # 10 periods of 1w = 20 * 12h periods
            
            if i < lookback_periods:
                signals[i] = 0.0
                continue
                
            # Get recent 1w high/low aligned to current 12h bar
            rolling_max_1w = pd.Series(high_1w).rolling(window=lookback_periods, min_periods=lookback_periods).max().values
            rolling_min_1w = pd.Series(low_1w).rolling(window=lookback_periods, min_periods=lookback_periods).min().values
            
            # Align to 12h timeframe
            donchian_high_aligned = align_htf_to_ltf(prices, df_1w, rolling_max_1w)
            donchian_low_aligned = align_htf_to_ltf(prices, df_1w, rolling_min_1w)
            
            if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
                np.isnan(volume_spike_1w_aligned[i])):
                signals[i] = 0.0
                continue
            
            # Look for Donchian breakout with volume confirmation
            if (high[i] >= donchian_high_aligned[i] and    # Break above upper band
                volume_spike_1w_aligned[i]):               # Volume confirmation
                position = 1
                entry_price = high[i]
                highest_since_entry = high[i]
                lowest_since_entry = high[i]  # Initialize for shorts
                signals[i] = 0.25
            elif (low[i] <= donchian_low_aligned[i] and    # Break below lower band
                  volume_spike_1w_aligned[i]):             # Volume confirmation
                position = -1
                entry_price = low[i]
                highest_since_entry = low[i]  # Initialize for longs
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals