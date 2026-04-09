#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume spike and volatility filter
# - Uses 4h Donchian channel (20-period high/low) for breakout signals
# - Confirms with 12h volume > 2.0x its 20-period average (strong participation)
# - Confirms with 12h ATR(14) > 1.5x its 50-period average (high volatility regime)
# - Uses ATR(14) trailing stop: exits when price retraces 3.0x ATR from extreme
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
# - Donchian channels adapt to volatility and provide objective breakout levels
# - Stricter filters reduce trade frequency while maintaining edge

name = "4h_12h_donchian_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h True Range for ATR
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h[0] = tr_12h[0]
    
    # 12h ATR(14) for volatility
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # 12h ATR(50) average for volatility regime filter
    atr_50_avg = pd.Series(atr_12h).rolling(window=50, min_periods=50).mean().values
    
    # 12h Volume > 2.0x 20-period average (stricter for fewer trades)
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = volume_12h > (2.0 * avg_volume_20)
    
    # Align 12h indicators to 4h
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    atr_50_avg_aligned = align_htf_to_ltf(prices, df_12h, atr_50_avg)
    volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h.astype(float))
    
    # 4h price data
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
        if (np.isnan(atr_12h_aligned[i]) or np.isnan(atr_50_avg_aligned[i]) or 
            np.isnan(volume_spike_12h_aligned[i]) or
            atr_12h_aligned[i] <= 0 or atr_50_avg_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 3.0x ATR from high (wider stop)
            if low[i] <= highest_since_entry - (3.0 * atr_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 3.0x ATR from low (wider stop)
            if high[i] >= lowest_since_entry + (3.0 * atr_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Calculate 4h Donchian channels using 20-period lookback
            # 20 periods of 4h = 10 periods of 12h (since 12h = 3 * 4h)
            lookback_periods = 10  # 10 periods of 12h = ~20 * 4h periods
            
            if i < lookback_periods:
                signals[i] = 0.0
                continue
                
            # Get recent 12h high/low aligned to current 4h bar
            rolling_max_12h = pd.Series(high_12h).rolling(window=lookback_periods, min_periods=lookback_periods).max().values
            rolling_min_12h = pd.Series(low_12h).rolling(window=lookback_periods, min_periods=lookback_periods).min().values
            
            # Align to 4h timeframe
            donchian_high_aligned = align_htf_to_ltf(prices, df_12h, rolling_max_12h)
            donchian_low_aligned = align_htf_to_ltf(prices, df_12h, rolling_min_12h)
            
            if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
                np.isnan(atr_12h_aligned[i]) or np.isnan(volume_spike_12h_aligned[i])):
                signals[i] = 0.0
                continue
            
            # Volatility regime filter: only trade when current ATR > 1.5x its 50-period average (stricter)
            volatility_filter = atr_12h_aligned[i] > (1.5 * atr_50_avg_aligned[i])
            
            # Look for Donchian breakout with volume and volatility confirmation
            if (high[i] >= donchian_high_aligned[i] and    # Break above upper band
                volume_spike_12h_aligned[i] and             # Volume confirmation
                volatility_filter):                        # High volatility regime
                position = 1
                entry_price = high[i]
                highest_since_entry = high[i]
                lowest_since_entry = high[i]  # Initialize for shorts
                signals[i] = 0.25
            elif (low[i] <= donchian_low_aligned[i] and    # Break below lower band
                  volume_spike_12h_aligned[i] and           # Volume confirmation
                  volatility_filter):                      # High volatility regime
                position = -1
                entry_price = low[i]
                highest_since_entry = low[i]  # Initialize for longs
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals