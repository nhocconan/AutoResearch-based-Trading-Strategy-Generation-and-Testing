#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian channels provide clear breakout levels, EMA50 on 1d ensures alignment with daily trend
# Volume confirmation filters false breakouts. Designed for 4h timeframe targeting 20-50 trades/year (75-200 total over 4 years)
# Uses discrete position sizing (0.30) to minimize fee churn and control drawdown
# Works in bull markets (breakout above upper + 1d EMA50 up) and bear markets (breakout below lower + 1d EMA50 down)

name = "4h_Donchian20_1dEMA50_Trend_Volume"
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
    
    # 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA calculation
        return np.zeros(n)
    
    # 1d EMA50 calculation
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period) - using 4h data
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(lookback, n):
        upper[i] = np.max(high[i-lookback:i])
        lower[i] = np.min(low[i-lookback:i])
    
    # Volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above upper Donchian with volume confirmation and uptrend
            if high[i] > upper[i-1] and volume_confirmation[i] and uptrend:
                signals[i] = 0.30
                position = 1
            # Short: Breakout below lower Donchian with volume confirmation and downtrend
            elif low[i] < lower[i-1] and volume_confirmation[i] and downtrend:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below lower Donchian (reversal) OR trend changes
            if low[i] < lower[i-1] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: Price breaks above upper Donchian (reversal) OR trend changes
            if high[i] > upper[i-1] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals