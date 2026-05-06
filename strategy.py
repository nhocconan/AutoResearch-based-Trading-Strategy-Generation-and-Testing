#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Donchian breakouts with volume confirmation and weekly trend filter
# Breakout above daily Donchian high(20) or below low(20) with volume > 1.5x 20-period average
# Trend filter: 50-period EMA on weekly timeframe to avoid counter-trend trades
# Works in bull/bear markets: breakouts capture trends, mean reversion within trend via stop
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_Donchian20_Volume_WeeklyTrend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily Donchian channels (20) ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily Donchian high and low (20-period)
    high_series = pd.Series(df_1d['high'])
    low_series = pd.Series(df_1d['low'])
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian levels to 12h timeframe
    dh_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    dl_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate weekly EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50)
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(dh_aligned[i]) or np.isnan(dl_aligned[i]) or 
            np.isnan(weekly_ema_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above daily Donchian high with volume confirmation and weekly uptrend
            if close[i] > dh_aligned[i] and volume_filter[i] and close[i] > weekly_ema_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below daily Donchian low with volume confirmation and weekly downtrend
            elif close[i] < dl_aligned[i] and volume_filter[i] and close[i] < weekly_ema_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below daily Donchian low (trailing stop)
            if close[i] < dl_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above daily Donchian high (trailing stop)
            if close[i] > dh_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals