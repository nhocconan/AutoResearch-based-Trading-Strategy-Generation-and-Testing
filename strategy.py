#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume and ATR regime filter
# - Uses 1d Donchian channel (20-period high/low) derived from actual 1d data for breakout signals
# - Confirms with 1w ATR(14) > 1.2x its 50-period average (high volatility regime)
# - Confirms with 1w volume > 1.5x its 20-period average (institutional participation)
# - Uses ATR(14) trailing stop: exits when price retraces 2.0x ATR from extreme
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years) to minimize fee drag
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
# - Donchian channels adapt to volatility and provide objective breakout levels

name = "1d_1w_donchian_atr_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # 1w ATR(14) for volatility
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # 1w ATR(50) average for volatility regime filter
    atr_50_avg = pd.Series(atr_1w).rolling(window=50, min_periods=50).mean().values
    
    # 1w Volume > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_spike_1w = volume_1w > (1.5 * avg_volume_20)
    
    # Align 1w indicators to 1d
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    atr_50_avg_aligned = align_htf_to_ltf(prices, df_1w, atr_50_avg)
    volume_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w.astype(float))
    
    # 1d price data
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
        if (np.isnan(atr_1w_aligned[i]) or np.isnan(atr_50_avg_aligned[i]) or 
            np.isnan(volume_spike_1w_aligned[i]) or
            atr_1w_aligned[i] <= 0 or atr_50_avg_aligned[i] <= 0):
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
            # Calculate 1d Donchian channels using 20-period lookback
            if i < 20:
                signals[i] = 0.0
                continue
                
            # Get recent 20-period high/low
            rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
            rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
            
            if (np.isnan(rolling_max[i]) or np.isnan(rolling_min[i]) or
                np.isnan(atr_1w_aligned[i]) or np.isnan(volume_spike_1w_aligned[i])):
                signals[i] = 0.0
                continue
            
            # Volatility regime filter: only trade when current ATR > 1.2x its 50-day average
            volatility_filter = atr_1w_aligned[i] > (1.2 * atr_50_avg_aligned[i])
            
            # Look for Donchian breakout with volume and volatility confirmation
            if (high[i] >= rolling_max[i] and    # Break above upper band
                volume_spike_1w_aligned[i] and   # Volume confirmation
                volatility_filter):              # High volatility regime
                position = 1
                entry_price = high[i]
                highest_since_entry = high[i]
                lowest_since_entry = high[i]  # Initialize for shorts
                signals[i] = 0.25
            elif (low[i] <= rolling_min[i] and    # Break below lower band
                  volume_spike_1w_aligned[i] and  # Volume confirmation
                  volatility_filter):             # High volatility regime
                position = -1
                entry_price = low[i]
                highest_since_entry = low[i]  # Initialize for longs
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals