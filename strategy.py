#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Weekly pivot direction (from prior week's close vs weekly EMA20) filters breakouts to trade with the weekly trend
# Volume confirmation avoids false breakouts. Designed for 6h timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# Uses discrete position sizing (0.25) to control drawdown and minimize fee churn
# Weekly trend filter ensures alignment with higher timeframe momentum, improving win rate in both bull and bear markets

name = "6h_Donchian20_WeeklyPivotDir_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot direction and Donchian calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for weekly EMA
        return np.zeros(n)
    
    # Weekly EMA20 for trend direction
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_close = df_1w['close'].values
    weekly_uptrend = weekly_close > ema_20_1w
    weekly_downtrend = weekly_close < ema_20_1w
    
    # Align weekly trend to 6h timeframe (wait for weekly bar to close)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # Daily data for Donchian(20) channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for Donchian calculation
        return np.zeros(n)
    
    # Calculate Donchian(20) channels from daily data
    highest_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 6h timeframe (wait for daily bar to close)
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # Volume confirmation on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)  # Strong volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or 
            np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend bias
        uptrend = weekly_uptrend_aligned[i] > 0.5
        downtrend = weekly_downtrend_aligned[i] > 0.5
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above Donchian(20) high with volume confirmation and weekly uptrend
            if high[i] > highest_20_aligned[i] and volume_confirmation[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below Donchian(20) low with volume confirmation and weekly downtrend
            elif low[i] < lowest_20_aligned[i] and volume_confirmation[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian(20) low (reversal) OR weekly trend changes to down
            if low[i] < lowest_20_aligned[i] or downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian(20) high (reversal) OR weekly trend changes to up
            if high[i] > highest_20_aligned[i] or uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals