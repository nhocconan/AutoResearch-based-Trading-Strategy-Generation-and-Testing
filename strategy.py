#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_WeeklyTrend_Filter_v1
Hypothesis: Daily Donchian(20) breakout with weekly EMA200 trend filter and volume confirmation.
- Primary timeframe: 1d for low trade frequency (target: 30-100 total trades over 4 years)
- Donchian channel breakout provides clear entry/exit signals
- Weekly EMA200 filter ensures alignment with higher timeframe trend (works in bull/bear)
- Volume confirmation (1.5x 20-day average) filters false breakouts
- ATR-based stoploss (2x ATR) manages risk
- Designed for 7-25 trades/year to minimize fee drag in ranging/bear markets (2025+)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for indicators
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Donchian channel (20-period) on daily data
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 1.5x 20-day average volume
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 20 for volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or
            np.isnan(vol_ma20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > period20_high[i-1]  # Break above previous period's high
        breakout_down = close[i] < period20_low[i-1]  # Break below previous period's low
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema200_1w_aligned[i]
        weekly_downtrend = close[i] < ema200_1w_aligned[i]
        
        if position == 0:
            # Long: breakout up AND volume spike AND weekly uptrend
            if breakout_up and volume_spike[i] and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: breakout down AND volume spike AND weekly downtrend
            elif breakout_down and volume_spike[i] and weekly_downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below Donchian low OR weekly trend turns down
            if close[i] < period20_low[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above Donchian high OR weekly trend turns up
            if close[i] > period20_high[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyTrend_Filter_v1"
timeframe = "1d"
leverage = 1.0