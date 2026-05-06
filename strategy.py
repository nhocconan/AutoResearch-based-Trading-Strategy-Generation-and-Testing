#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout with volume confirmation and trend filter
# Weekly Donchian channels provide robust support/resistance levels that work across market regimes
# Breakout above weekly high or below weekly low with volume > 1.5x 20-day average indicates strong momentum
# Trend filter: 50-period EMA on daily timeframe to avoid counter-trend trades
# Works in bull/bear markets: breakouts capture trends, reversals capture pullbacks within trend
# Target: 30-100 total trades over 4 years (7-25/year) with 0.25 position sizing

name = "1d_WeeklyDonchian20_VolumeTrendFilter_v1"
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
    
    # Calculate weekly Donchian channels ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    weekly_high = df_1w['high'].rolling(window=20, min_periods=20).max().values
    weekly_low = df_1w['low'].rolling(window=20, min_periods=20).min().values
    
    # Align weekly levels to daily timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Volume confirmation: >1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Trend filter: 50-period EMA on daily timeframe
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend = close > ema_50
    downtrend = close < ema_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly high with volume confirmation and uptrend
            if close[i] > weekly_high_aligned[i] and volume_filter[i] and uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below weekly low with volume confirmation and downtrend
            elif close[i] < weekly_low_aligned[i] and volume_filter[i] and downtrend[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly low (failed support) or reaches opposite band (take profit)
            if close[i] < weekly_low_aligned[i] or close[i] > weekly_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly high (failed resistance) or reaches opposite band (take profit)
            if close[i] > weekly_high_aligned[i] or close[i] < weekly_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals