#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA200 trend filter and volume confirmation.
# Uses price channel breakouts for trend following, filtered by higher-timeframe trend and volume.
# Designed to capture strong trends while avoiding whipsaws in ranging markets.
# Works in bull markets (buy breakouts above upper band) and bear markets (sell breakdowns below lower band).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Donchian channels (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d data for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === 4h volume confirmation ===
    vol_avg20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    warmup = 200  # For EMA200
    position = 0
    
    for i in range(warmup, n):
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_avg20[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        vol_filter = volume[i] > 1.5 * vol_avg20[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + 1d uptrend + volume
            if close[i] > highest_high[i] and close[i] > ema200_1d_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below lower Donchian + 1d downtrend + volume
            if close[i] < lowest_low[i] and close[i] < ema200_1d_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long on breakdown below lower Donchian or trend reversal
            if close[i] < lowest_low[i] or close[i] < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short on breakout above upper Donchian or trend reversal
            if close[i] > highest_high[i] or close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA200_VolumeFilter"
timeframe = "4h"
leverage = 1.0