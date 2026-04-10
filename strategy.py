#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ATR filter and weekly volume confirmation
# - Long when price breaks above Donchian(20) high on 6h chart with 1d ATR(14) > 1.5x 50-period average (high volatility regime)
# - Short when price breaks below Donchian(20) low on 6h chart with same volatility filter
# - Weekly volume spike (>2.0x 20-period average) confirms institutional participation
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Volatility filter reduces false breakouts in low-volume ranging markets
# - Weekly trend alignment ensures trading with higher timeframe momentum

name = "6h_1d_1w_donchian_breakout_volatility_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d ATR for volatility filter
    high_low = df_1d['high'] - df_1d['low']
    high_close = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    low_close = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_ranges = np.nanmax(ranges.values, axis=1)
    atr_14_1d = pd.Series(true_ranges).rolling(window=14, min_periods=14).mean().values
    atr_ma_50_1d = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_ma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50_1d)
    
    # Pre-compute 1w volume confirmation
    volume_1w = df_1w['volume'].values
    avg_volume_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_spike_1w = volume_1w > (2.0 * avg_volume_20_1w)
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w)
    
    # Pre-compute 6h Donchian channels
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    donchian_high_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_ma_50_1d_aligned[i]) or
            np.isnan(vol_spike_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when price closes below Donchian low
            if prices['close'].iloc[i] < donchian_low_20[i]:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price closes above Donchian high
            if prices['close'].iloc[i] > donchian_high_20[i]:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volatility filter: current ATR > 1.5x 50-period average ATR
            volatility_filter = atr_14_1d_aligned[i] > (1.5 * atr_ma_50_1d_aligned[i])
            
            # Long signal: price breaks above Donchian high with volatility filter and weekly volume spike
            if (prices['high'].iloc[i] > donchian_high_20[i] and 
                volatility_filter and 
                vol_spike_1w_aligned[i]):
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = 0.25
            # Short signal: price breaks below Donchian low with volatility filter and weekly volume spike
            elif (prices['low'].iloc[i] < donchian_low_20[i] and 
                  volatility_filter and 
                  vol_spike_1w_aligned[i]):
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = -0.25
    
    return signals