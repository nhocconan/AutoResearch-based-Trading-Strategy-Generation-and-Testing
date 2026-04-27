#!/usr/bin/env python3
"""
Hypothesis: 6-hour Williams Fractal breakout with daily trend filter and volume confirmation.
Uses daily Williams Fractals as potential reversal points, entering on breakouts
in the direction of the daily EMA trend. Volume filter ensures only significant
breakouts are traded. Designed to work in both bull and bear markets by using
the daily trend as filter.
Target: 12-37 trades/year per symbol (48-148 total over 4 years) to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Fractals and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Williams Fractals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractal: bearish = high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    #             bullish  = low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    n1d = len(high_1d)
    bearish_fractal = np.full(n1d, np.nan)
    bullish_fractal = np.full(n1d, np.nan)
    
    for i in range(2, n1d - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] < high_1d[i-1] and
            high_1d[i-3] < high_1d[i-1] and
            high_1d[i+1] < high_1d[i-1]):
            bearish_fractal[i] = high_1d[i]
        
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] > low_1d[i-1] and
            low_1d[i-3] > low_1d[i-1] and
            low_1d[i+1] > low_1d[i-1]):
            bullish_fractal[i] = low_1d[i]
    
    # Williams Fractals need 2 extra daily bars for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Get 6-hour data for volume filter
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 6-hour volume MA(20)
    vol_6h = df_6h['volume'].values
    vol_ma_20_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Williams Fractals, volume MA, and daily EMA
    start_idx = max(34, 20)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_ma_20_6h_aligned[i]) or np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        bearish_fractal_level = bearish_fractal_aligned[i]
        bullish_fractal_level = bullish_fractal_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_6h_aligned[i]
        trend_1d = ema_34_1d_aligned[i]
        
        # Volume filter: volume > 2.0x 6h average (strict to reduce trades)
        vol_filter = vol_now > 2.0 * vol_ma
        
        # Entry conditions: Williams Fractal breakout with volume and daily trend alignment
        if position == 0:
            # Long: break above bullish fractal + volume + daily uptrend
            if not np.isnan(bullish_fractal_level) and close[i] > bullish_fractal_level and vol_filter and close[i] > trend_1d:
                signals[i] = size
                position = 1
            # Short: break below bearish fractal + volume + daily downtrend
            elif not np.isnan(bearish_fractal_level) and close[i] < bearish_fractal_level and vol_filter and close[i] < trend_1d:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below daily EMA
            if close[i] < trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above daily EMA
            if close[i] > trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsFractal_Breakout_DailyTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0