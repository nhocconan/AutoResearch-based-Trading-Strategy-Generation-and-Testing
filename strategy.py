#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and ATR-based trailing stop
# - Uses 12h Donchian channel (20-period high/low) for breakout signals
# - Confirms with 1d volume > 1.8x 20-period average (institutional participation)
# - Uses ATR(14) trailing stop: exits when price retraces 2.5x ATR from extreme
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 12-25 trades/year on 12h timeframe (48-100 total over 4 years) to minimize fee drag
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
# - Donchian channels adapt to volatility and provide objective breakout levels

name = "12h_1d_donchian_volume_atr_v1"
timeframe = "12h"
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
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr_1d[0]
    
    # 1d ATR(14) for volatility
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1d Volume > 1.8x 20-period average
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.8 * avg_volume_20)
    
    # Align 1d indicators to 12h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or
            atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.5x ATR from high OR volume spike reversal
            if low[i] <= highest_since_entry - (2.5 * atr_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.5x ATR from low OR volume spike reversal
            if high[i] >= lowest_since_entry + (2.5 * atr_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Calculate 12h Donchian channels using 12-period lookback
            # We need at least 20 periods of 12h data for Donchian(20)
            # Since we don't have direct 12h HTF data, we'll use 1d data as proxy
            # and calculate Donchian from 1d high/low with appropriate lookback
            
            # For 12h timeframe, we approximate Donchian(20) using 1d data
            # 20 periods of 12h = 10 periods of 1d (since 1d = 2 * 12h)
            lookback_periods = 10  # 10 days = ~20 * 12h periods
            
            if i < lookback_periods:
                signals[i] = 0.0
                continue
                
            # Get recent 1d high/low aligned to current 12h bar
            # We'll use the rolling max/min of 1d high/low over lookback period
            # and align to 12h timeframe
            
            # Calculate 1d rolling max/min for Donchian channels
            rolling_max_1d = pd.Series(high_1d).rolling(window=lookback_periods, min_periods=lookback_periods).max().values
            rolling_min_1d = pd.Series(low_1d).rolling(window=lookback_periods, min_periods=lookback_periods).min().values
            
            # Align to 12h timeframe
            donchian_high_aligned = align_htf_to_ltf(prices, df_1d, rolling_max_1d)
            donchian_low_aligned = align_htf_to_ltf(prices, df_1d, rolling_min_1d)
            
            if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i])):
                signals[i] = 0.0
                continue
            
            # Look for Donchian breakout with volume confirmation
            if (high[i] >= donchian_high_aligned[i] and  # Break above upper band
                volume_spike_1d_aligned[i]):            # Volume confirmation
                position = 1
                entry_price = high[i]
                atr_stop = atr_1d_aligned[i]
                highest_since_entry = high[i]
                lowest_since_entry = high[i]  # Initialize for shorts
                signals[i] = 0.25
            elif (low[i] <= donchian_low_aligned[i] and   # Break below lower band
                  volume_spike_1d_aligned[i]):            # Volume confirmation
                position = -1
                entry_price = low[i]
                atr_stop = atr_1d_aligned[i]
                highest_since_entry = low[i]  # Initialize for longs
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals