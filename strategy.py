# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly price channels (Donchian) for trend direction,
# confirmed by daily volume spike and weekly momentum (ROC). The strategy enters long
# when price breaks above the weekly Donchian upper channel with volume confirmation
# and weekly ROC > 0, and short when price breaks below the weekly Donchian lower
# channel with volume confirmation and weekly ROC < 0. Uses a weekly Donchian channel
# for trend filtering to avoid counter-trend trades in choppy markets.
# Target: 10-20 trades/year per symbol with disciplined entries to minimize fee drag.
name = "1d_WeeklyDonchian20_Volume_ROC"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channel (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channel on weekly data
    donchian_high = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Weekly ROC for momentum confirmation
    roc_1w = pd.Series(df_1w['close']).pct_change(periods=1).values  # 1-week ROC
    roc_1w_aligned = align_htf_to_ltf(prices, df_1w, roc_1w)
    
    # Daily volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(roc_1w_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high, with volume spike and positive weekly ROC
            if (close[i] > donchian_high_aligned[i] and 
                volume_spike[i] and 
                roc_1w_aligned[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low, with volume spike and negative weekly ROC
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_spike[i] and 
                  roc_1w_aligned[i] < 0):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly Donchian low or weekly ROC turns negative
            if (close[i] < donchian_low_aligned[i]) or (roc_1w_aligned[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly Donchian high or weekly ROC turns positive
            if (close[i] > donchian_high_aligned[i]) or (roc_1w_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals