#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout + volume confirmation + session filter (08-20 UTC)
# Uses 4h Donchian channels (20-period) for trend direction, with 1h breakout confirmation and volume spike
# Designed for 1h timeframe to target 15-35 trades/year per symbol.
# Works in bull/bear via multi-timeframe trend alignment and volume-based breakout signals.
# Entry only during active market hours (08-20 UTC) to avoid low-volume noise.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for Donchian channels (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # 1h volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.8 * vol_ma20  # Moderate threshold for balanced trade frequency
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma20[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian high + volume spike
            if close[i] > donchian_high_aligned[i] and vol_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian low + volume spike
            elif close[i] < donchian_low_aligned[i] and vol_spike[i]:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: price returns to middle of 4h Donchian channel
            donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if position == 1:
                if close[i] < donchian_mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if close[i] > donchian_mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_DonchianBreakout_4hVolSession"
timeframe = "1h"
leverage = 1.0