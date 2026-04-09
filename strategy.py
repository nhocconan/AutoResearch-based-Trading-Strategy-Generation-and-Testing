#!/usr/bin/env python3
# 1d_weekly_donchian_breakout_volume_v2
# Hypothesis: 1d strategy using weekly Donchian channels (20) for breakout signals with volume confirmation and weekly EMA trend filter.
# Long: Price breaks above weekly Donchian H20, weekly close > weekly EMA20, daily volume > 1.5x 20-day average.
# Short: Price breaks below weekly Donchian L20, weekly close < weekly EMA20, daily volume > 1.5x 20-day average.
# Exit: Price crosses weekly EMA20 or opposite Donchian breakout.
# Uses weekly structure for major trend, volume confirmation to filter weak breakouts, EMA20 for trend alignment.
# Target: 7-25 trades/year (30-100 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_volume_v2"
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
    
    # Get weekly data for Donchian channels and EMA (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate rolling max/min for Donchian channels
    high_series = pd.Series(high_1w)
    low_series = pd.Series(low_1w)
    donchian_h = high_series.rolling(window=20, min_periods=20).max().values
    donchian_l = low_series.rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Align HTF indicators to daily timeframe (wait for completed weekly bar)
    donchian_h_aligned = align_htf_to_ltf(prices, df_1w, donchian_h)
    donchian_l_aligned = align_htf_to_ltf(prices, df_1w, donchian_l)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Daily volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_h_aligned[i]) or np.isnan(donchian_l_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price crosses below weekly EMA20 OR breaks below weekly Donchian L20
            if close[i] < ema20_1w_aligned[i] or low[i] < donchian_l_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above weekly EMA20 OR breaks above weekly Donchian H20
            if close[i] > ema20_1w_aligned[i] or high[i] > donchian_h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above weekly Donchian H20, close above weekly EMA20, volume confirmed
            if (high[i] > donchian_h_aligned[i] and close[i] > ema20_1w_aligned[i] and volume_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below weekly Donchian L20, close below weekly EMA20, volume confirmed
            elif (low[i] < donchian_l_aligned[i] and close[i] < ema20_1w_aligned[i] and volume_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals