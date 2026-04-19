#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1-week trend filter and volume confirmation.
# Long when: price breaks above 20-day Donchian upper, weekly close > weekly open, volume > 1.5x 20-day average
# Short when: price breaks below 20-day Donchian lower, weekly close < weekly open, volume > 1.5x 20-day average
# Exit when price returns to the opposite Donchian band (mean reversion) or weekly trend reverses.
# Daily timeframe captures major moves, weekly filter ensures alignment with larger trend.
# Target: 10-25 trades/year per symbol. Low frequency reduces fee drag.
name = "1d_Donchian20_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Weekly bullish/bearish trend (1 = bullish week, -1 = bearish week)
    weekly_trend = np.where(close_1w > open_1w, 1, -1)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # 20-day Donchian channels
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-day volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(weekly_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        weekly_trend_val = weekly_trend_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian upper, weekly bullish, volume confirmation
            if price > donchian_upper[i] and weekly_trend_val == 1 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower, weekly bearish, volume confirmation
            elif price < donchian_lower[i] and weekly_trend_val == -1 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to Donchian lower (mean reversion) or weekly trend turns bearish
            if price < donchian_lower[i] or weekly_trend_val == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to Donchian upper (mean reversion) or weekly trend turns bullish
            if price > donchian_upper[i] or weekly_trend_val == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals