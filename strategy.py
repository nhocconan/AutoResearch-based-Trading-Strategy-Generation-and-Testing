#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily price action filtered by weekly Bollinger Band squeeze and Donchian breakout.
# Long when price breaks above upper Donchian(20) with weekly BB width < 30th percentile (low volatility),
# short when breaks below lower Donchian(20) under same conditions.
# Uses volatility contraction (BB squeeze) followed by expansion (breakout) to capture trends in both bull and bear markets.
# Designed for low trade frequency (~10-20/year) to minimize fee decay on 1d timeframe.

name = "1d_1w_bb_squeeze_donchian_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Bollinger Bands (20, 2)
    close_1w = df_1w['close'].values
    bb_mid = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Weekly BB width percentile (30-day lookback for squeeze threshold)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=30, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    # Squeeze condition: BB width < 30th percentile (low volatility)
    squeeze_condition = bb_width_percentile < 30
    squeeze_aligned = align_htf_to_ltf(prices, df_1w, squeeze_condition)
    
    # Daily Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 50 to ensure all indicators are valid
    for i in range(50, n):
        # Skip if squeeze data is invalid
        if np.isnan(squeeze_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volatility contraction filter: weekly BB squeeze
        vol_filter = squeeze_aligned[i]
        
        # Breakout conditions: price breaks Donchian bands with volatility contraction
        long_breakout = high[i] > donch_high[i]
        short_breakout = low[i] < donch_low[i]
        
        long_entry = long_breakout and vol_filter
        short_entry = short_breakout and vol_filter
        
        # Exit: reverse Donchian breakout
        exit_long = low[i] < donch_low[i]
        exit_short = high[i] > donch_high[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals