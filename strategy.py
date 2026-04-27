# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Hypothesis: 1-day Donchian channel breakout with weekly volume confirmation and monthly trend filter.
Trades breakouts above/below the 1-day Donchian(20) when weekly volume exceeds the 1-week average and the monthly trend aligns.
Designed to work in both bull and bear markets by using monthly trend as filter and weekly volume to confirm breakout strength.
Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day Donchian(20) channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Get weekly data for volume filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1-week volume MA(20)
    vol_1w = df_1w['volume'].values
    vol_ma_20_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    # Get monthly data for trend filter
    df_1m = get_htf_data(prices, '1M')
    if len(df_1m) < 50:
        return np.zeros(n)
    
    # Calculate monthly EMA(50) for trend
    close_1m = df_1m['close'].values
    ema_50_1m = pd.Series(close_1m).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1m_aligned = align_htf_to_ltf(prices, df_1m, ema_50_1m)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Donchian, volume MA, and monthly EMA
    start_idx = max(20, 20, 50)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_20_1w_aligned[i]) or 
            np.isnan(ema_50_1m_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Current daily price and volume
        high_1d_now = high_1d[i] if i < len(high_1d) else high_1d[-1]
        low_1d_now = low_1d[i] if i < len(low_1d) else low_1d[-1]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1w_aligned[i]
        trend_1m = ema_50_1m_aligned[i]
        
        # Donchian breakout conditions
        upper_break = high_1d_now > donchian_high_aligned[i]
        lower_break = low_1d_now < donchian_low_aligned[i]
        
        # Volume filter: volume > 1.5x 1-week average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: Donchian breakout with volume and monthly trend alignment
        if position == 0:
            # Long: breakout above upper Donchian + volume + monthly uptrend
            if upper_break and vol_filter and close[i] > trend_1m:
                signals[i] = size
                position = 1
            # Short: breakout below lower Donchian + volume + monthly downtrend
            elif lower_break and vol_filter and close[i] < trend_1m:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian midline or monthly trend turns down
            midline = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if close[i] < midline or close[i] < trend_1m:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to Donchian midline or monthly trend turns up
            midline = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if close[i] > midline or close[i] > trend_1m:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_Volume_1wTrendFilter_1mTrend"
timeframe = "1d"
leverage = 1.0