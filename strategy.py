#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Donchian breakout with volume confirmation and 1d volatility filter.
# Long when price breaks above 1w Donchian high (20-period) with volume > 1.5x average and 1d ATR > 1d ATR MA.
# Short when price breaks below 1w Donchian low (20-period) with volume > 1.5x average and 1d ATR > 1d ATR MA.
# Exit when price returns to 1w Donchian midpoint or volatility drops below threshold.
# Designed to capture strong trends in both bull and bear markets with volatility filtering to avoid chop.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Calculate 1w Donchian channels (20-period)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Load 1d data ONCE for volatility filter
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
    atr_ma_1d = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    
    # Align indicators to lower timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # Need Donchian and ATR
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(atr_ma_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Volatility filter: current ATR > ATR MA
        volatility_filter = atr_1d_aligned[i] > atr_ma_1d_aligned[i]
        
        if position == 0:
            # Look for Donchian breakouts
            # Long: price breaks above Donchian high with volume and volatility
            if (close[i] > donchian_high_aligned[i] and 
                volume_confirmed and 
                volatility_filter):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low with volume and volatility
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_confirmed and 
                  volatility_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian mid or volatility drops
            if (close[i] <= donchian_mid_aligned[i] or 
                not volatility_filter):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian mid or volatility drops
            if (close[i] >= donchian_mid_aligned[i] or 
                not volatility_filter):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_DonchianBreakout_VolumeVolatilityFilter_v1"
timeframe = "12h"
leverage = 1.0