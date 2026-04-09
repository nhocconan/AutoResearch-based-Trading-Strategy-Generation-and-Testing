#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# - Uses 4h Donchian channel breakout (20-period) for trend entry
# - Regime filter: only trade when 1d ATR(14) > 20-period SMA of ATR (high volatility regime)
# - Volume confirmation: 4h volume > 1.5x its 20-period average
# - ATR trailing stop: exit when price retraces 2.0x ATR from extreme
# - Position size: 0.25 (25% of capital)
# - Target: 30-60 trades/year on 4h timeframe (120-240 total over 4 years)
# - Donchian breakouts work in both trending and volatile markets
# - ATR regime filter avoids low-volatility choppy markets where breakouts fail
# - Volume confirmation reduces false breakouts
# - ATR stop manages risk during volatile periods

name = "4h_1d_donchian_atr_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d True Range for ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr_1d[0]
    
    # 1d ATR(14)
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1d ATR SMA(20) for regime filter
    atr_sma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # 1d High Volatility Regime: ATR > SMA of ATR
    high_vol_regime = atr_1d > atr_sma_20
    
    # 4h Donchian channel (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian upper (20-period high)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian lower (20-period low)
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # 4h volume > 1.5x 20-period average (volume confirmation)
    volume_4h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume_4h > (1.5 * avg_volume_20)
    
    # Align 1d indicators to 4h
    high_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, high_vol_regime)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(high_vol_regime_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(volume_spike_4h[i]) or
            atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high_4h[i] > highest_since_entry:
                highest_since_entry = high_4h[i]
            
            # Exit conditions: price retraces 2.0x ATR from high
            if low_4h[i] <= highest_since_entry - (2.0 * atr_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low_4h[i] < lowest_since_entry:
                lowest_since_entry = low_4h[i]
            
            # Exit conditions: price retraces 2.0x ATR from low
            if high_4h[i] >= lowest_since_entry + (2.0 * atr_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and volatility regime
            if (high_4h[i] >= donchian_high[i] and    # Break above Donchian high
                volume_spike_4h[i] and                # Volume confirmation
                high_vol_regime_aligned[i]):          # High volatility regime
                position = 1
                entry_price = high_4h[i]
                highest_since_entry = high_4h[i]
                lowest_since_entry = high_4h[i]  # Initialize for shorts
                signals[i] = 0.25
            elif (low_4h[i] <= donchian_low[i] and    # Break below Donchian low
                  volume_spike_4h[i] and              # Volume confirmation
                  high_vol_regime_aligned[i]):        # High volatility regime
                position = -1
                entry_price = low_4h[i]
                highest_since_entry = low_4h[i]  # Initialize for longs
                lowest_since_entry = low_4h[i]
                signals[i] = -0.25
    
    return signals