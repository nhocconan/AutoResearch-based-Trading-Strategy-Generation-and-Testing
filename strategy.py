#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation (>1.5x 20-bar MA) + 12h EMA50 trend filter
# Long when price breaks above Donchian upper channel with volume spike and price > 12h EMA50
# Short when price breaks below Donchian lower channel with volume spike and price < 12h EMA50
# Uses discrete sizing (0.25) to minimize fee churn. Target: 75-200 total trades over 4 years.

name = "4h_Donchian20_Breakout_VolumeSpike_12hEMA50_Trend_v1"
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
    
    # 12h HTF data for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) on 12h close
    ema_12h_50 = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 4h timeframe
    ema_12h_50_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # Donchian channels (20-period)
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_12h_50_aligned[i]) or np.isnan(high_ma_20[i]) or 
            np.isnan(low_ma_20[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian upper channel, volume spike, price > 12h EMA50
            if curr_close > high_ma_20[i] and vol_spike and curr_close > ema_12h_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower channel, volume spike, price < 12h EMA50
            elif curr_close < low_ma_20[i] and vol_spike and curr_close < ema_12h_50_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price below Donchian lower channel
            if curr_close < low_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price above Donchian upper channel
            if curr_close > high_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals