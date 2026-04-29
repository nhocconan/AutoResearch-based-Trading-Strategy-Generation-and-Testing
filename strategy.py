#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA50 trend filter + volume spike confirmation
# Donchian breakouts capture strong momentum moves; 1d EMA50 filters for higher-timeframe trend alignment
# Volume spike confirms institutional participation. Works in bull markets via breakout continuation
# and in bear markets via mean-reversion failures at channel edges. Target: 12-30 trades/year (50-120 total).

name = "12h_Donchian20_Breakout_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 12h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (1.5 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 20, 30)  # warmup for 1d EMA, Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        
        # Determine trend direction from 1d EMA50
        # Uptrend: price > EMA50, Downtrend: price < EMA50
        is_uptrend = curr_close > curr_ema_50_1d
        is_downtrend = curr_close < curr_ema_50_1d
        
        if position == 0:  # Flat - look for new entries
            # Long: breakout above upper Donchian + uptrend + volume confirmation
            if curr_high > curr_highest_high and is_uptrend and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below lower Donchian + downtrend + volume confirmation
            elif curr_low < curr_lowest_low and is_downtrend and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price crosses below EMA50 OR hits lower Donchian (mean reversion)
            if curr_close < curr_ema_50_1d or curr_low <= curr_lowest_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price crosses above EMA50 OR hits upper Donchian (mean reversion)
            if curr_close > curr_ema_50_1d or curr_high >= curr_highest_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals