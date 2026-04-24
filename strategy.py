#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1w pivot direction filter and volume confirmation.
- Uses 6h primary timeframe with 1d HTF for Camarilla levels and 1w HTF for pivot bias
- Long when: price breaks above H3 (6h) AND 1w pivot shows bullish bias (close > weekly pivot) AND volume > 2.0 * 20-period average
- Short when: price breaks below L3 (6h) AND 1w pivot shows bearish bias (close < weekly pivot) AND volume > 2.0 * 20-period average
- Exit when: price returns to 6h pivot level (PP) OR volume drops below average
- Camarilla levels provide precise intraday support/resistance; weekly pivot filters for higher timeframe bias; volume confirms breakout strength
- Designed to work in both bull (breakouts with weekly bias) and bear (breakdowns with weekly bias) markets
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h Camarilla levels (H3, L3, PP)
    lookback = 20
    rolling_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    rolling_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    rolling_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().values
    
    # Camarilla calculations
    rang = rolling_high - rolling_low
    h3 = rolling_close + (rang * 1.1 / 4)
    l3 = rolling_close - (rang * 1.1 / 4)
    pp = (rolling_high + rolling_low + rolling_close) / 3
    
    # Calculate 1d volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    # Calculate 1w pivot for trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly OHLC for pivot calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point
    pp_1w = (high_1w + low_1w + close_1w) / 3
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or np.isnan(pp[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(pp_1w_aligned[i]) or
            np.isnan(close_1w[-1] if len(close_1w) > 0 else np.nan)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current weekly close for bias (use last available weekly value)
        # Find the most recent completed weekly bar
        weekly_idx = min(len(close_1w) - 1, i // (7 * 24 * 60 // 360))  # approximate weekly bars
        if weekly_idx < 0:
            weekly_idx = 0
        weekly_close = close_1w[weekly_idx] if weekly_idx < len(close_1w) else close_1w[-1]
        
        if position == 0:
            # Long: price breaks above H3 AND weekly close > weekly PP AND volume confirmation
            if close[i] > h3[i] and weekly_close > pp_1w[weekly_idx] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND weekly close < weekly PP AND volume confirmation
            elif close[i] < l3[i] and weekly_close < pp_1w[weekly_idx] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to 6h PP OR volume drops below average
            if close[i] <= pp[i] or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to 6h PP OR volume drops below average
            if close[i] >= pp[i] or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_1wPivot_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0