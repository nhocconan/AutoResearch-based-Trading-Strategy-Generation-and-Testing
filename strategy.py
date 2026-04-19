#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian(20) breakout with volume confirmation and 12h trend filter
# - 1d Donchian(20) upper/lower channels define breakout levels
# - 12h EMA(34) defines trend direction (long when close > EMA34, short when close < EMA34)
# - Volume filter: 1d volume > 1.5x 20-period average for conviction
# - Entry: long when price breaks above 1d Donchian upper + trend up + volume
#          short when price breaks below 1d Donchian lower + trend down + volume
# - Exit: opposite Donchian breakout or trend reversal
# - Position size: 0.25 to balance risk and reward
# - Designed for fewer trades (target ~25-40/year) to minimize fee drag
# - Works in bull/bear by following higher timeframe trend and requiring volume confirmation

name = "4h_Donchian1d_EMA34_Volume_v1"
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
    
    # Get 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Donchian(20) channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(34) for trend direction
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_34_12h_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 1.5x average
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Look for long entry: price > 1d Donchian high + trend up + volume
            if close[i] > donchian_high_aligned[i] and close[i] > ema_34_12h_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price < 1d Donchian low + trend down + volume
            elif close[i] < donchian_low_aligned[i] and close[i] < ema_34_12h_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on price < 1d Donchian low or trend reversal
            if close[i] < donchian_low_aligned[i] or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on price > 1d Donchian high or trend reversal
            if close[i] > donchian_high_aligned[i] or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals