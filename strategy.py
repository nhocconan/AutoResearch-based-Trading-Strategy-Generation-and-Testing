#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with volume confirmation and weekly trend filter.
# Strategy buys on breakout above 20-day high with volume > 20-day average and weekly close > weekly open,
# sells on breakout below 20-day low with volume > 20-day average and weekly close < weekly open.
# Uses tight entry conditions to limit trades (target: 50-100 total over 4 years) and avoid fee drag.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue).
# Risk managed via time-based exit: exit after 5 days if no breakout in opposite direction.

name = "1d_donchian20_volume_weekly_trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly trend: bullish if weekly close > weekly open, bearish if close < open
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open
    weekly_bearish = weekly_close < weekly_open
    
    # Align weekly trend to daily (shifted by 1 week for prior week's close)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Volume confirmation: 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    days_in_trade = 0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(weekly_bearish_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
                days_in_trade += 1
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: require volume above 20-day average
        vol_filter = volume[i] > vol_ma[i]
        
        # Update days in trade
        if position != 0:
            days_in_trade += 1
        
        # Exit conditions
        if position == 1:  # long position
            # Exit: breakdown below Donchian low OR max 5 days in trade
            if close[i] < donchian_low[i] or days_in_trade >= 5:
                signals[i] = 0.0
                position = 0
                days_in_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: breakout above Donchian high OR max 5 days in trade
            if close[i] > donchian_high[i] or days_in_trade >= 5:
                signals[i] = 0.0
                position = 0
                days_in_trade = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume filter and weekly trend alignment
            if vol_filter:
                # Long: breakout above Donchian high with weekly bullish bias
                if close[i] > donchian_high[i] and weekly_bullish_aligned[i] > 0.5:
                    signals[i] = 0.25
                    position = 1
                    days_in_trade = 1
                # Short: breakdown below Donchian low with weekly bearish bias
                elif close[i] < donchian_low[i] and weekly_bearish_aligned[i] > 0.5:
                    signals[i] = -0.25
                    position = -1
                    days_in_trade = 1
    
    return signals