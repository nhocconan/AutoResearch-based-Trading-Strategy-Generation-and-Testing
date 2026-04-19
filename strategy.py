#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with daily pivot reversal filter and volume confirmation
# - 6h Donchian(20) breakout: long when price breaks above 20-period high, short when breaks below 20-period low
# - Daily pivot reversal filter: only take long if price < daily pivot point, short if price > daily pivot point
#   This filters breakouts that go against the intraday bias, improving win rate in ranging markets
# - Volume confirmation: 6h volume > 1.5x 20-period average volume to ensure conviction
# - Designed to work in both bull and bear markets by following price action with volume confirmation
# - Target: 15-30 trades/year (~60-120 total over 4 years) to avoid excessive fee drag

name = "6h_DonchianBreakout_1dPivotReversal_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Daily pivot point: (H + L + C) / 3
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    pivot_1d = typical_price_1d.values
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Volume confirmation: 20-period average volume on 6h
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(pivot_1d_aligned[i]) or np.isnan(vol_ma_6h[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 6h volume > 1.5x 20-period average
        volume_filter = vol_ma_6h[i] > 0 and volume[i] > 1.5 * vol_ma_6h[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high AND price < daily pivot (bullish bias)
            if close[i] > donchian_high[i] and close[i] < pivot_1d_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND price > daily pivot (bearish bias)
            elif close[i] < donchian_low[i] and close[i] > pivot_1d_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on Donchian low break or reversal above pivot
            if close[i] < donchian_low[i] or close[i] > pivot_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on Donchian high break or reversal below pivot
            if close[i] > donchian_high[i] or close[i] < pivot_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals