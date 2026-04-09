#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume and volatility regime filter
# - Uses 4h Donchian channel (20-period high/low) derived from 1d data for breakout signals
# - Confirms with 1d volume > 1.8x its 20-period average (strong institutional participation)
# - Confirms with 1d ATR(14) > 1.3x its 50-period average (high volatility regime)
# - Uses ATR(14) trailing stop: exits when price retraces 2.5x ATR from extreme
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 20-40 trades/year on 4h timeframe (80-160 total over 4 years) to minimize fee drag
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
# - Donchian channels adapt to volatility and provide objective breakout levels

name = "4h_1d_donchian_atr_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
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
    
    # 1d ATR(50) average for volatility regime filter
    atr_50_avg = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # 1d Volume > 1.8x 20-period average (stricter for fewer trades)
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.8 * avg_volume_20)
    
    # Align 1d indicators to 4h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_50_avg_aligned = align_htf_to_ltf(prices, df_1d, atr_50_avg)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(atr_50_avg_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or
            atr_1d_aligned[i] <= 0 or atr_50_avg_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.5x ATR from high (wider stop)
            if low[i] <= highest_since_entry - (2.5 * atr_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.5x ATR from low (wider stop)
            if high[i] >= lowest_since_entry + (2.5 * atr_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Calculate 4h Donchian channels using 20-period lookback
            # 20 periods of 4h = 10 periods of 1d (since 1d = 6 * 4h)
            lookback_periods = 10  # 10 days = ~20 * 4h periods
            
            if i < lookback_periods:
                signals[i] = 0.0
                continue
                
            # Get recent 1d high/low aligned to current 4h bar
            rolling_max_1d = pd.Series(high_1d).rolling(window=lookback_periods, min_periods=lookback_periods).max().values
            rolling_min_1d = pd.Series(low_1d).rolling(window=lookback_periods, min_periods=lookback_periods).min().values
            
            # Align to 4h timeframe
            donchian_high_aligned = align_htf_to_ltf(prices, df_1d, rolling_max_1d)
            donchian_low_aligned = align_htf_to_ltf(prices, df_1d, rolling_min_1d)
            
            if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
                np.isnan(atr_1d_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
                signals[i] = 0.0
                continue
            
            # Volatility regime filter: only trade when current ATR > 1.3x its 50-day average (stricter)
            volatility_filter = atr_1d_aligned[i] > (1.3 * atr_50_avg_aligned[i])
            
            # Look for Donchian breakout with volume and volatility confirmation
            if (high[i] >= donchian_high_aligned[i] and    # Break above upper band
                volume_spike_1d_aligned[i] and             # Volume confirmation
                volatility_filter):                        # High volatility regime
                position = 1
                entry_price = high[i]
                highest_since_entry = high[i]
                lowest_since_entry = high[i]  # Initialize for shorts
                signals[i] = 0.25
            elif (low[i] <= donchian_low_aligned[i] and    # Break below lower band
                  volume_spike_1d_aligned[i] and           # Volume confirmation
                  volatility_filter):                      # High volatility regime
                position = -1
                entry_price = low[i]
                highest_since_entry = low[i]  # Initialize for longs
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals