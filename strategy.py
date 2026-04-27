#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian channel breakout (20-period) with volume confirmation and daily trend filter.
Trades only on breakouts in the direction of the daily EMA trend, using volume > 1.5x average as confirmation.
Designed to work in both bull and bear markets by using the daily trend as filter.
Target: 20-50 trades/year per symbol (80-200 total over 4 years) to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4-hour Donchian channels (20-period)
    high_4h = get_htf_data(prices, '4h')['high'].values
    low_4h = get_htf_data(prices, '4h')['low'].values
    upper_donchian = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_donchian = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4-hour volume average (20-period)
    vol_4h = get_htf_data(prices, '4h')['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all 4h indicators
    upper_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '4h'), upper_donchian)
    lower_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '4h'), lower_donchian)
    vol_ma_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '4h'), vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Donchian channels, volume MA, and daily EMA
    start_idx = max(20, 20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        upper_band = upper_aligned[i]
        lower_band = lower_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_aligned[i]
        trend_1d = ema_34_1d_aligned[i]
        
        # Volume filter: volume > 1.5x 4h average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: Donchian breakout with volume and daily trend alignment
        if position == 0:
            # Long: break above upper band + volume + daily uptrend
            if close[i] > upper_band and vol_filter and close[i] > trend_1d:
                signals[i] = size
                position = 1
            # Short: break below lower band + volume + daily downtrend
            elif close[i] < lower_band and vol_filter and close[i] < trend_1d:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below daily EMA or lower Donchian band
            if close[i] < trend_1d or close[i] < lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above daily EMA or upper Donchian band
            if close[i] > trend_1d or close[i] > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_DonchianBreakout_Volume_DailyTrendFilter"
timeframe = "4h"
leverage = 1.0