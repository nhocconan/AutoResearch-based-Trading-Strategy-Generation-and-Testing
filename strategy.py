#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian channel breakout (20) with 1d EMA34 trend filter and volume spike.
Breakouts above/below 20-period Donchian channels capture momentum. EMA34 on 1d filters for
trend alignment to avoid counter-trend trades. Volume confirmation ensures breakout validity.
Designed for 12h timeframe to target 12-37 trades/year, minimizing fee drag while working
in bull/bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for EMA34 trend filter and Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d Donchian channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: volume / 20-period average volume (1d)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_34_1d_aligned[i]
        upper_channel = donchian_high_aligned[i]
        lower_channel = donchian_low_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        vol_threshold = 1.5  # Volume must be 1.5x average
        
        if position == 0:
            # Enter long: price breaks above Donchian upper, volume spike, uptrend
            if (price_close > upper_channel and 
                vol_ratio > vol_threshold and 
                price_close > ema_trend):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian lower, volume spike, downtrend
            elif (price_close < lower_channel and 
                  vol_ratio > vol_threshold and 
                  price_close < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to opposite Donchian band or trend reversal
            if position == 1 and (price_close < lower_channel or price_close < ema_trend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > upper_channel or price_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_DonchianBreakout_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0