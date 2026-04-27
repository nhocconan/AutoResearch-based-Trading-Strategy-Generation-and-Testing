#!/usr/bin/env python3
"""
1d_Kelly_Donchian_Breakout_1wTrend_Volume
Hypothesis: Trade 1d timeframe with weekly Donchian breakout filtered by 1w trend and volume spikes.
Breakouts above/below weekly Donchian(20) indicate strong momentum, filtered by 1w EMA trend and volume.
Designed for 15-25 trades/year to minimize fee drift. Works in bull via breakouts and bear via breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channels (20-period high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    donchian_high = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Weekly EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: current volume > 2.5 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Donchian and volume average
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        ema50_val = ema_50_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Require minimum 5 days since last exit to avoid churn
            if bars_since_exit >= 5:
                # Long: price breaks above weekly Donchian high with volume confirmation AND above weekly EMA50 (uptrend)
                if close[i] > donch_high and vol_conf and close[i] > ema50_val:
                    signals[i] = size
                    position = 1
                    bars_since_exit = 0
                # Short: price breaks below weekly Donchian low with volume confirmation AND below weekly EMA50 (downtrend)
                elif close[i] < donch_low and vol_conf and close[i] < ema50_val:
                    signals[i] = -size
                    position = -1
                    bars_since_exit = 0
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low (opposite level)
            if close[i] < donch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high (opposite level)
            if close[i] > donch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Kelly_Donchian_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0