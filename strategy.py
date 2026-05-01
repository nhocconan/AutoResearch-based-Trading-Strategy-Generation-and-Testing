#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume spike (>2.0x 20-bar MA)
# Donchian breakout captures momentum, 1d EMA34 filters for higher timeframe trend (long when price > EMA34, short when price < EMA34),
# volume spike confirms strong participation. Works in bull markets via breakouts above EMA34 and in bear markets via breakdowns below EMA34.
# Discrete sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "4h_Donchian20_1dEMA34_VolumeSpike_v1"
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
    
    # 1d HTF data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend direction
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian(20) channels on 4h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(lookback, 20, 34)  # Need 20 for Donchian/volume MA, 34 for EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper band, above 1d EMA34, and volume spike
            if curr_close > highest_high[i-1] and curr_close > ema_34_1d_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band, below 1d EMA34, and volume spike
            elif curr_close < lowest_low[i-1] and curr_close < ema_34_1d_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price breaking below Donchian lower band or below 1d EMA34
            if curr_close < lowest_low[i-1] or curr_close < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price breaking above Donchian upper band or above 1d EMA34
            if curr_close > highest_high[i-1] or curr_close > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals