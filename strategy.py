#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily Donchian breakout with volume confirmation and volatility filter.
# Breakouts above/below Donchian(20) from prior day trigger entries, filtered by volume spike and
# ATR-based volatility regime. Works in bull/bear by capturing momentum after volatility expansion.
# Target: 20-50 trades/year per symbol (<200 total over 4 years).
name = "4h_DailyDonchian20_VolumeVolatilityFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels and ATR (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Donchian channels (20-period)
    # Upper band = max(high, lookback=20), Lower band = min(low, lookback=20)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Volume filter: volume > 1.5 * 20-period average (on 4h data)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    # Volatility filter: current ATR > 0.8 * 50-period average ATR (avoid low volatility)
    atr_ma = pd.Series(atr_aligned).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr_aligned > (atr_ma * 0.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for ATR MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above daily Donchian upper band
            if close[i] > donchian_upper_aligned[i] and volume_filter[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below daily Donchian lower band
            elif close[i] < donchian_lower_aligned[i] and volume_filter[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit: price re-enters Donchian channel (middle) or volatility drops
            if close[i] < donchian_upper_aligned[i] or not volatility_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: price re-enters Donchian channel (middle) or volatility drops
            if close[i] > donchian_lower_aligned[i] or not volatility_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals