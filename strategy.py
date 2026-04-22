#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4-hour Donchian breakout with 12-hour trend filter and volume confirmation
    # Donchian(20) breakout captures momentum; 12h EMA50 ensures trend alignment
    # Volume spike confirms institutional participation
    # Targets ~25-35 trades/year to minimize fee drag while capturing strong trends
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels (20-period) on 4h
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike filter (20-period on 4h)
    vol_ma20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > 2.0 * vol_ma20
    
    # Align indicators to 4h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_4h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_4h, low_20)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    vol_spike_aligned = align_htf_to_ltf(prices, df_4h, vol_spike)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper Donchian + price above 12h EMA50 + volume spike
            if close[i] > high_20_aligned[i] and close[i] > ema50_12h_aligned[i] and vol_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian + price below 12h EMA50 + volume spike
            elif close[i] < low_20_aligned[i] and close[i] < ema50_12h_aligned[i] and vol_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back through Donchian middle or trend reverses
            if position == 1:
                if close[i] < (high_20_aligned[i] + low_20_aligned[i]) / 2 or close[i] < ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > (high_20_aligned[i] + low_20_aligned[i]) / 2 or close[i] > ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian_20_Breakout_12hEMA50_Volume_Spike_v1"
timeframe = "4h"
leverage = 1.0