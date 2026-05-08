#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Donchian_Breakout_Trend_Filter_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily volume filter
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above weekly Donchian high + trend filter + volume
            long_cond = (close[i] > donchian_high_20_aligned[i] and
                        close[i] > ema50_1w_aligned[i] and
                        volume[i] > vol_ma20[i])
            
            # Short: break below weekly Donchian low + trend filter + volume
            short_cond = (close[i] < donchian_low_20_aligned[i] and
                         close[i] < ema50_1w_aligned[i] and
                         volume[i] > vol_ma20[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below weekly Donchian low OR trend reversal
            exit_cond = (close[i] < donchian_low_20_aligned[i] or
                        close[i] < ema50_1w_aligned[i])
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above weekly Donchian high OR trend reversal
            exit_cond = (close[i] > donchian_high_20_aligned[i] or
                        close[i] > ema50_1w_aligned[i])
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Donchian breakout with trend filter and volume confirmation.
# Enters long when price breaks above weekly 20-period Donchian high with weekly EMA50 uptrend and volume confirmation.
# Enters short when price breaks below weekly 20-period Donchian low with weekly EMA50 downtrend and volume confirmation.
# Exits on breakdown below weekly Donchian low (for longs) or breakout above weekly Donchian high (for shorts),
# or when trend reverses (price crosses weekly EMA50).
# Designed for 1d timeframe to target 20-50 trades over 4 years (5-12/year) to minimize fee drag.
# Uses discrete sizing (0.25) to reduce churn. Works in both bull (trend following breaks) and bear (trend following breakdowns).