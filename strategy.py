#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Donchian_Breakout_VolumeTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for Donchian channels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate upper and lower bands
    upper_1w = np.full(len(high_1w), np.nan)
    lower_1w = np.full(len(low_1w), np.nan)
    
    for i in range(len(high_1w)):
        if i >= 19:  # 20-period lookback
            upper_1w[i] = np.max(high_1w[i-19:i+1])
            lower_1w[i] = np.min(low_1w[i-19:i+1])
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to daily timeframe
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily volume confirmation (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_1w_aligned[i]) or np.isnan(lower_1w_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Price breaks above weekly upper Donchian + above weekly EMA50 + volume spike
            long_cond = (close[i] > upper_1w_aligned[i]) and (close[i] > ema50_1w_aligned[i]) and (volume[i] > vol_ma20[i] * 1.5)
            
            # Short entry: Price breaks below weekly lower Donchian + below weekly EMA50 + volume spike
            short_cond = (close[i] < lower_1w_aligned[i]) and (close[i] < ema50_1w_aligned[i]) and (volume[i] > vol_ma20[i] * 1.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses below weekly lower Donchian OR loses trend (below EMA50)
            if (close[i] < lower_1w_aligned[i]) or (close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above weekly upper Donchian OR gains trend (above EMA50)
            if (close[i] > upper_1w_aligned[i]) or (close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Donchian breakouts with volume confirmation and trend filter capture strong momentum moves.
# Long when price breaks above weekly 20-period high AND above weekly EMA50 with volume confirmation.
# Short when price breaks below weekly 20-period low AND below weekly EMA50 with volume confirmation.
# Exits when price reverses back below the opposite Donchian band or loses trend alignment.
# Works in bull markets (captures breakouts) and bear markets (captures breakdowns).
# Weekly timeframe reduces noise and false signals compared to lower timeframes.
# Volume confirmation ensures breakouts are supported by participation.
# Target: 20-60 total trades over 4 years = 5-15/year to minimize fee decay while capturing major moves.