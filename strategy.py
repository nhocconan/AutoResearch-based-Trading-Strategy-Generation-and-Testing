#!/usr/bin/env python3
name = "1d_Weekly_Donchian_20_Breakout_Trend"
timeframe = "1d"
leverage = 1.0

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
    
    # Weekly Donchian channels (20-bar high/low) for trend filter
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # 20-period Donchian upper/lower bounds
    donchian_upper = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_weekly, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_weekly, donchian_lower)
    
    # Daily ATR for volatility filter and stop loss
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    tr = np.concatenate([[0], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily volume confirmation
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure warm-up for weekly indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close breaks above weekly Donchian upper + volume confirmation
            if (close[i] > donchian_upper_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below weekly Donchian lower + volume confirmation
            elif (close[i] < donchian_lower_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below weekly Donchian lower or ATR-based stop
            if (close[i] < donchian_lower_aligned[i] or 
                close[i] < (prices['close'].iloc[i-1] - 2.0 * atr[i]) if i > 0 else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above weekly Donchian upper or ATR-based stop
            if (close[i] > donchian_upper_aligned[i] or 
                close[i] > (prices['close'].iloc[i-1] + 2.0 * atr[i]) if i > 0 else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals