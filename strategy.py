#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Squeeze + Volume Breakout with 1-week trend filter.
# Long when: BB Width at 6-month low + price breaks above upper BB + volume > 2x average + weekly close > weekly open
# Short when: BB Width at 6-month low + price breaks below lower BB + volume > 2x average + weekly close < weekly open
# Bollinger Squeeze identifies low volatility periods preceding explosive moves.
# Volume confirmation ensures breakout validity. Weekly trend filter aligns with higher timeframe momentum.
# Designed to work in both bull (breakouts up) and bear (breakdowns) markets.
# Target: 20-40 trades/year per symbol.
name = "6h_BollingerSqueeze_VolumeBreakout_WeeklyTrend"
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
    
    # 6h Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = ma + bb_std * std
    lower = ma - bb_std * std
    bb_width = upper - lower
    
    # 6h BB Width percentile (lookback ~6 months = 20*4*6 = 480 periods)
    bb_width_percentile = pd.Series(bb_width).rolling(window=480, min_periods=100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # 6h Volume spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    weekly_bullish = weekly_close > weekly_open  # True if weekly close > open
    
    # Align weekly trend to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 480, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_width_percentile[i]) or np.isnan(ma[i]) or 
            np.isnan(std[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(weekly_bullish_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Bollinger Squeeze condition: BB Width at extreme low (bottom 10%)
        is_squeeze = bb_width_percentile[i] <= 10
        
        # Breakout conditions
        breakout_up = close[i] > upper[i]
        breakout_down = close[i] < lower[i]
        
        # Volume confirmation: volume > 2x average
        volume_spike = volume[i] > 2 * vol_ma[i]
        
        # Weekly trend filter
        weekly_bull = weekly_bullish_aligned[i] > 0.5
        
        if position == 0:
            # Long entry: squeeze + upward breakout + volume + weekly bullish
            if is_squeeze and breakout_up and volume_spike and weekly_bull:
                signals[i] = 0.25
                position = 1
            # Short entry: squeeze + downward breakout + volume + weekly bearish
            elif is_squeeze and breakout_down and volume_spike and not weekly_bull:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below middle band (mean reversion) or squeeze breaks down
            if close[i] < ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above middle band (mean reversion) or squeeze breaks up
            if close[i] > ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals