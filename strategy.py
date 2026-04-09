#!/usr/bin/env python3
# 6h_volatility_breakout_volume_v2
# Hypothesis: 6h strategy using volatility expansion (ATR-based breakout) from 1d HTF for trend context,
# combined with 6h price breaking Donchian channels and volume confirmation.
# Long: 1d ATR expansion + price breaks above 6h Donchian(20) high + volume spike
# Short: 1d ATR expansion + price breaks below 6h Donchian(20) low + volume spike
# Uses discrete sizing (±0.25) to minimize fee churn. Target: 75-200 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_volatility_breakout_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for ATR-based volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ATR(14) for volatility measurement
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1d ATR ratio: current ATR / 20-period average ATR (volatility expansion filter)
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio_1d = atr_1d / atr_ma_1d
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # 6h Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility expansion filter: ATR ratio > 1.5 (expanding volatility)
        vol_expansion = atr_ratio_aligned[i] > 1.5
        
        if position == 1:  # Long position
            # Exit: volatility contraction OR price retracement to midpoint
            midpoint = (donch_high[i] + donch_low[i]) / 2
            if not vol_expansion or close[i] < midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: volatility contraction OR price retracement to midpoint
            midpoint = (donch_high[i] + donch_low[i]) / 2
            if not vol_expansion or close[i] > midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need both volatility expansion and volume confirmation
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if vol_expansion and volume_confirmed:
                # Long: price breaks above Donchian high
                if close[i] > donch_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low
                elif close[i] < donch_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals