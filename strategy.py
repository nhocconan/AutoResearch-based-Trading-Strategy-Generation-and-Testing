#!/usr/bin/env python3
"""
1d Donchian Channel Breakout with Weekly EMA Trend Filter and Volume Spike
Hypothesis: On the daily timeframe, Donchian(20) breakouts capture significant price moves.
Weekly EMA34 filter ensures we only trade in the direction of the higher timeframe trend,
reducing false signals during sideways markets. Volume spike confirms institutional participation.
This strategy works in both bull and bear markets by trading breakouts in the weekly trend direction.
Target: 15-25 trades/year on 1d timeframe (60-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend direction
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian Channel (20-period) on daily timeframe
    # Upper band = highest high over past 20 days
    # Lower band = lowest low over past 20 days
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.5 * 20-day average
    volume_series = pd.Series(volume)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators (Donchian 20 + weekly EMA 34 + volume MA 20)
    start_idx = max(20, 34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Weekly trend filter: price above/below weekly EMA34
        weekly_uptrend = curr_close > ema_34_1w_aligned[i]
        weekly_downtrend = curr_close < ema_34_1w_aligned[i]
        
        # Donchian breakout conditions
        breakout_long = curr_high > donchian_upper[i-1]  # Break above previous period's upper band
        breakout_short = curr_low < donchian_lower[i-1]  # Break below previous period's lower band
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + weekly trend alignment + volume spike
            long_entry = breakout_long and weekly_uptrend and vol_spike
            short_entry = breakout_short and weekly_downtrend and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price retouches Donchian lower band (mean reversion within the channel)
            if curr_close < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price retouches Donchian upper band (mean reversion within the channel)
            if curr_close > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0