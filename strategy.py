#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h trend filter and volume confirmation.
# Uses 12h Donchian channel for trend direction (more stable than EMA) to avoid whipsaw.
# Trades only when 6h price breaks above/below 20-period Donchian channel with volume spike.
# 12h Donchian trend filter ensures we trade with the intermediate-term trend.
# Volume confirmation (>2.0x 20-period average) filters low-momentum breakouts.
# Works in bull (buy upper breakout with uptrend) and bear (sell lower breakdown with downtrend).
# Discrete position sizing 0.25 balances return and drawdown. Target: 50-150 trades over 4 years.

name = "6h_Donchian20_Breakout_12hTrend_VolumeConfirm_v1"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Donchian channel (20-period) for trend filter
    # Upper = max(high, 20), Lower = min(low, 20)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    donchian_20_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_20_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # 12h trend: price above/below midpoint of Donchian channel
    donchian_midpoint = (donchian_20_high + donchian_20_low) / 2.0
    donchian_20_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_20_high)
    donchian_20_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_20_low)
    donchian_midpoint_aligned = align_htf_to_ltf(prices, df_12h, donchian_midpoint)
    
    # 6h Donchian breakout levels (20-period)
    donchian_6h_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_6h_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume on 6h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(20, 20) + 1  # 21 (for Donchian channels and volume MA20)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_20_high_aligned[i]) or 
            np.isnan(donchian_20_low_aligned[i]) or
            np.isnan(donchian_midpoint_aligned[i]) or
            np.isnan(donchian_6h_high[i]) or
            np.isnan(donchian_6h_low[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # 12h trend filter: price relative to Donchian midpoint
        uptrend = curr_close > donchian_midpoint_aligned[i]
        downtrend = curr_close < donchian_midpoint_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 6h Donchian breakout conditions
        breakout_upper = curr_close > donchian_6h_high[i]  # Break above upper band
        breakdown_lower = curr_close < donchian_6h_low[i]  # Break below lower band
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above upper band AND uptrend AND volume confirmation
            if breakout_upper and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below lower band AND downtrend AND volume confirmation
            elif breakdown_lower and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on breakdown below lower band (reversal signal)
            if curr_close < donchian_6h_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on breakout above upper band (reversal signal)
            if curr_close > donchian_6h_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals