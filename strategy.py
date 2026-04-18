#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout + Volume Spike + 1w EMA Trend Filter
Long when price breaks above weekly Donchian high with volume spike and price > 1w EMA50.
Short when price breaks below weekly Donchian low with volume spike and price < 1w EMA50.
Designed for low trade frequency (<25/year) with trend-following edge in bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian and EMA (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        above_donchian_high = price > donchian_high_aligned[i]
        below_donchian_low = price < donchian_low_aligned[i]
        above_1w_ema = price > ema_50_1w_aligned[i]
        below_1w_ema = price < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high, price above 1w EMA, volume spike
            if (above_donchian_high and above_1w_ema and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low, price below 1w EMA, volume spike
            elif (below_donchian_low and below_1w_ema and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: break below Donchian low or price falls below 1w EMA
            if below_donchian_low or below_1w_ema:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: break above Donchian high or price rises above 1w EMA
            if above_donchian_high or above_1w_ema:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume_Spike_EMA50"
timeframe = "1d"
leverage = 1.0