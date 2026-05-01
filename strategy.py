#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA50 trend filter + volume spike (>2x 20-bar MA)
# Donchian breakout captures strong momentum moves. 1d EMA50 ensures alignment with higher-timeframe trend to avoid counter-trend whipsaws.
# Volume spike confirms institutional participation. Discrete sizing (0.25) minimizes fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits.
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.

name = "12h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) on 1d close
    ema_1d_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 12h timeframe
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Donchian channels (20-period) on 12h data
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_50_aligned[i]) or np.isnan(high_ma_20[i]) or 
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
            # Long: price breaks above Donchian upper channel, above 1d EMA50, volume spike
            if curr_high > high_ma_20[i] and curr_close > ema_1d_50_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower channel, below 1d EMA50, volume spike
            elif curr_low < low_ma_20[i] and curr_close < ema_1d_50_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price below Donchian lower channel or below 1d EMA50
            if curr_low < low_ma_20[i] or curr_close < ema_1d_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price above Donchian upper channel or above 1d EMA50
            if curr_high > high_ma_20[i] or curr_close > ema_1d_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals