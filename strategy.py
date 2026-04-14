#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Donchian breakout with 1d volatility filter and volume confirmation.
# Long when price breaks above 1w Donchian upper band with 1d ATR > 1d ATR MA (volatility expansion) and volume > 1.5x 20-period average.
# Short when price breaks below 1w Donchian lower band with same volatility and volume conditions.
# Exit when price returns to 1w midline or volatility contracts (ATR < ATR MA).
# Designed to capture strong trending moves while avoiding choppy markets.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period) on 1w
    donchian_window = 20
    donchian_high = pd.Series(high_1w).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low_1w).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Load 1d data ONCE for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR on 1d
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_period = 14
    atr_1d = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    atr_ma = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Volatility expansion: current ATR > ATR moving average
    vol_expansion = atr_1d > atr_ma
    
    # Align indicators to lower timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    vol_expansion_aligned = align_htf_to_ltf(prices, df_1d, vol_expansion)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(donchian_window, 20)  # Need Donchian and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or
            np.isnan(vol_expansion_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for Donchian breakouts
            # Long: price breaks above Donchian high AND volatility expansion
            if (close[i] > donchian_high_aligned[i] and 
                vol_expansion_aligned[i] and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low AND volatility expansion
            elif (close[i] < donchian_low_aligned[i] and 
                  vol_expansion_aligned[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian midline or volatility contracts
            if (close[i] <= donchian_mid_aligned[i] or 
                not vol_expansion_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian midline or volatility contracts
            if (close[i] >= donchian_mid_aligned[i] or 
                not vol_expansion_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1wDonchian_1dVol_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0