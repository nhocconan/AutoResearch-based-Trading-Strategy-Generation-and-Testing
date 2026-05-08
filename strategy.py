#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h trading with 4h Donchian breakout + volume confirmation + 1d trend filter.
# Uses 4h Donchian channels for breakout signals, volume spike for confirmation, and 1d EMA50 for trend filter.
# Long when price breaks above 4h upper band + volume > 1.5x average + price > 1d EMA50.
# Short when price breaks below 4h lower band + volume > 1.5x average + price < 1d EMA50.
# Includes session filter (08-20 UTC) to avoid low-volatility periods.
# Target: 60-150 total trades over 4 years (15-37/year) to balance opportunity and fee drag.
# Works in bull (breakout continuation) and bear (breakout reversals from range).

name = "1h_DonchianBreakout_4hVol_1dTrend"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 4h Donchian channels (20-period)
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 1h
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(in_session[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h upper band, volume spike, price > 1d EMA50
            if (close[i] > upper_20_aligned[i] and vol_ratio[i] > 1.5 and close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h lower band, volume spike, price < 1d EMA50
            elif (close[i] < lower_20_aligned[i] and vol_ratio[i] > 1.5 and close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below 4h lower band or volume drops
            if (close[i] < lower_20_aligned[i] or vol_ratio[i] < 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above 4h upper band or volume drops
            if (close[i] > upper_20_aligned[i] or vol_ratio[i] < 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals