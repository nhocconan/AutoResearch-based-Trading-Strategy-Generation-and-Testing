#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 1-week Donchian breakout with 1-day volume confirmation
# - Long when price breaks above weekly Donchian high (20-period) + daily volume > 1.5x 20-day average
# - Short when price breaks below weekly Donchian low (20-period) + daily volume > 1.5x 20-day average
# - Exit when price returns to weekly Donchian midpoint
# - Designed to capture strong weekly trends with volume confirmation, avoiding false breakouts
# - Target: 12-30 trades/year to minimize fee drag on 12h timeframe
name = "12h_WeeklyDonchian_1dVolume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian channels (20-period)
    donch_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donch_mid_1w = (donch_high_1w + donch_low_1w) / 2.0
    
    # Align weekly Donchian levels to 12h timeframe (wait for weekly close)
    donch_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_high_1w)
    donch_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_low_1w)
    donch_mid_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_mid_1w)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    
    # Daily volume average (20-period)
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for weekly and daily indicators
    
    for i in range(start_idx, n):
        if np.isnan(donch_high_1w_aligned[i]) or np.isnan(donch_low_1w_aligned[i]) or np.isnan(donch_mid_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current daily volume > 1.5x 20-day average
        vol_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above weekly Donchian high + volume confirmation
            if close[i] > donch_high_1w_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly Donchian low + volume confirmation
            elif close[i] < donch_low_1w_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to weekly Donchian midpoint
            if close[i] >= donch_mid_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly Donchian midpoint
            if close[i] <= donch_mid_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals