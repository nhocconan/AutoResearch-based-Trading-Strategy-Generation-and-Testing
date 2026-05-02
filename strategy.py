#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses 1w HTF for EMA50 to capture weekly trend and reduce false breakouts in choppy markets.
# Donchian(20) from 12h provides proven price channel breakouts with good historical performance.
# Volume confirmation at 2.0x average ensures strong participation while limiting trades (~12-37/year target).
# Discrete sizing 0.25 to minimize fee churn. Works in bull/bear: trend filter ensures trades only with momentum.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag.

name = "12h_Donchian20_Breakout_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) levels from 12h timeframe (using prior completed 12h bar)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Prior 20 completed 12h bars for Donchian calculation
    high_12h = df_12h['high'].rolling(window=20, min_periods=20).max().shift(1).values
    low_12h = df_12h['low'].rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, high_12h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, low_12h)
    
    # 1w EMA50 for trend filter (weekly trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 51:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: 2.0x 20-period average (strict threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian high AND price > 1w EMA50 AND volume spike
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND price < 1w EMA50 AND volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below Donchian low OR price < 1w EMA50
            if close[i] < donchian_low_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above Donchian high OR price > 1w EMA50
            if close[i] > donchian_high_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals