#!/usr/bin/env python3
"""
4h_12h_donchian_volume_trend_v1
Strategy: 4h Donchian breakout with 12h trend filter and volume confirmation
Timeframe: 4h
Leverage: 1.0
Hypothesis: Combines 4h Donchian channel breakout (20-period) with 12h EMA trend filter and volume expansion (1.5x 20-period average) to capture strong momentum moves. Designed for moderate trade frequency (20-40/year) to balance signal quality and cost efficiency. Works in both bull and bear markets by requiring trend alignment and volume confirmation, reducing false breakouts during choppy periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_donchian_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 4h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h EMA trend filter (21-period)
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Session filter: 08-20 UTC (major sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(ema_12h_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        ema_trend = ema_12h_aligned[i]
        
        # Volume confirmation: 4h volume must be expanded (1.5x 20-period average)
        volume_expanded = volume_current > 1.5 * vol_ma
        
        # Long conditions: price breaks above Donchian high + volume expansion + above 12h EMA trend
        long_signal = volume_expanded and (price_close > donchian_high[i]) and (price_close > ema_trend)
        
        # Short conditions: price breaks below Donchian low + volume expansion + below 12h EMA trend
        short_signal = volume_expanded and (price_close < donchian_low[i]) and (price_close < ema_trend)
        
        # Exit when price returns to the midpoint of the Donchian channel
        donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
        exit_long = position == 1 and price_close < donchian_mid
        exit_short = position == -1 and price_close > donchian_mid
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals