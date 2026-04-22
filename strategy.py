#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Works in bull/bear by using 1d EMA34 for trend direction + volume spike for momentum confirmation.
# Target: 20-50 trades/year (80-200 total over 4 years) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Load 4h data for Donchian calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper/lower bands (previous 20 bars, not including current)
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().shift(1).values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align to 4h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    
    # Calculate 4h volume average (20-period)
    vol_4h = df_4h['volume'].values
    vol_avg_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or np.isnan(vol_avg_20_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian with uptrend and volume
            if (close[i] > upper_20_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.8 * vol_avg_20_4h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian with downtrend and volume
            elif (close[i] < lower_20_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.8 * vol_avg_20_4h_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to the opposite Donchian level
            if position == 1:
                if close[i] < lower_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > upper_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_Donchian20_1dEMA34_Volume_Breakout"
timeframe = "4h"
leverage = 1.0