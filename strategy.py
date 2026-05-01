#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 12h EMA50 trend filter + volume spike (>2.0x 20-bar volume MA)
# Donchian breakout captures strong momentum; 12h EMA50 ensures alignment with higher-timeframe trend to avoid counter-trend trades.
# Volume spike (>2.0x) confirms institutional participation, reducing false breakouts. Discrete sizing (0.25) minimizes fee churn.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull (breakouts with volume) and bear (failed reversals reverse quickly).

name = "6h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) on 12h close
    ema_12h_50 = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 6h timeframe
    ema_12h_50_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # Donchian Channel (20-period) on 6h data
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 50 for 12h EMA and 20 for Donchian (50 > 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_12h_50_aligned[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Donchian breakout conditions
        breakout_up = curr_close > highest_high_20[i-1]  # Break above prior period high
        breakout_down = curr_close < lowest_low_20[i-1]  # Break below prior period low
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up, price above 12h EMA50, volume spike
            if breakout_up and curr_close > ema_12h_50_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down, price below 12h EMA50, volume spike
            elif breakout_down and curr_close < ema_12h_50_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Donchian breakdown or price below 12h EMA50
            if curr_close < lowest_low_20[i] or curr_close < ema_12h_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Donchian breakout or price above 12h EMA50
            if curr_close > highest_high_20[i] or curr_close > ema_12h_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals