#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1-week trend filter and volume confirmation
# Long when price breaks above 20-day high, weekly trend is up (price > weekly EMA50), and volume > 1.5x average
# Short when price breaks below 20-day low, weekly trend is down (price < weekly EMA50), and volume > 1.5x average
# Uses discrete position sizing (0.25) to limit turnover and manage drawdown in both bull and bear markets
# Targets 30-100 total trades over 4 years (7-25/year) to avoid fee drag while capturing meaningful moves

name = "1d_Donchian20_WeeklyTrend_Volume"
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
    
    # Get 1d data for Donchian channels (20-period high/low)
    # Note: Since we're on 1d timeframe, we can calculate directly but must ensure no look-ahead
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        upper_break = donchian_high[i]
        lower_break = donchian_low[i]
        weekly_trend = ema50_1w_aligned[i]
        vol_ok = vol_confirm[i]
        
        if position == 0:
            # Enter long: price breaks above 20-day high, weekly trend up, volume confirmation
            if close_val > upper_break and weekly_trend > 0 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 20-day low, weekly trend down, volume confirmation
            elif close_val < lower_break and weekly_trend < 0 and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below 20-day low or weekly trend turns down
            if close_val < lower_break or weekly_trend < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above 20-day high or weekly trend turns up
            if close_val > upper_break or weekly_trend > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals