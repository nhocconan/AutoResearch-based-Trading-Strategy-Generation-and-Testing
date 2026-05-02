#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Donchian breakouts capture institutional participation at key support/resistance
# Weekly pivot direction (from 1w data) filters breakouts to align with higher timeframe trend
# Volume confirmation ensures breakouts have conviction
# Designed for 6h timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Works in bull markets (upward weekly pivot + Donchian breakout) and bear markets (downward weekly pivot + Donchian breakdown)

name = "6h_Donchian20_1wPivot_Dir_Volume"
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
    
    # 1w data for pivot direction (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need enough for pivot calculation
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard floor trader pivots)
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # We use the weekly pivot P as trend indicator: price > P = up-trend, price < P = down-trend
    weekly_p = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3.0
    weekly_p_aligned = align_htf_to_ltf(prices, df_1w, weekly_p)
    
    # Donchian channels (20-period) on 6h data
    # Upper channel = highest high over past 20 periods
    # Lower channel = lowest low over past 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.8 * vol_ema_20)  # Moderate threshold for volume spike
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Donchian channels)
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_p_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from weekly pivot (price relative to weekly pivot)
        weekly_uptrend = close[i] > weekly_p_aligned[i]
        weekly_downtrend = close[i] < weekly_p_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above Donchian upper with volume confirmation and weekly uptrend
            if close[i] > donchian_upper[i] and weekly_uptrend and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below Donchian lower with volume confirmation and weekly downtrend
            elif close[i] < donchian_lower[i] and weekly_downtrend and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian lower (reversal) OR weekly trend turns down
            if close[i] < donchian_lower[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian upper (reversal) OR weekly trend turns up
            if close[i] > donchian_upper[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals