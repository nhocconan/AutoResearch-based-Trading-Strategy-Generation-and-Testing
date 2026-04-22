#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend and Donchian channel (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly Donchian channel (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    upper_donch_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_donch_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly indicators to daily
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    upper_donch_20_aligned = align_htf_to_ltf(prices, df_1w, upper_donch_20)
    lower_donch_20_aligned = align_htf_to_ltf(prices, df_1w, lower_donch_20)
    
    # Daily volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_donch_20_aligned[i]) or 
            np.isnan(lower_donch_20_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above weekly EMA50 and breaks above weekly upper Donchian + volume spike
            if close[i] > ema_50_1w_aligned[i] and close[i] > upper_donch_20_aligned[i] and volume[i] > 2.0 * vol_avg_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly EMA50 and breaks below weekly lower Donchian + volume spike
            elif close[i] < ema_50_1w_aligned[i] and close[i] < lower_donch_20_aligned[i] and volume[i] > 2.0 * vol_avg_20[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back below/above weekly EMA50 (full exit)
            if position == 1:
                # Exit long: Price closes below weekly EMA50
                if close[i] < ema_50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price closes above weekly EMA50
                if close[i] > ema_50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_WeeklyEMA50_Donchian20_Breakout_Volume_Spike"
timeframe = "1d"
leverage = 1.0