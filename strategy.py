#!/usr/bin/env python3
# 12h_Donchian_Breakout_1dTrend_VolumeConfirmation
# Hypothesis: On 12h timeframe, enter long when price breaks above Donchian upper band with price > daily EMA50 and volume spike.
# Enter short when price breaks below Donchian lower band with price < daily EMA50 and volume spike.
# Exit when price crosses below/above Donchian midpoint or daily EMA50.
# Uses daily timeframe for trend filter to avoid counter-trend trades.
# Aims for 15-30 trades/year to minimize fee drag and works in both bull and bear markets by following higher-timeframe trend.

name = "12h_Donchian_Breakout_1dTrend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) on 12h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Load daily data for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_volume = df_1d['volume'].values
    
    # Daily EMA50 for trend filter
    daily_ema50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume 20-period moving average for confirmation
    daily_vol_ma = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 12h timeframe
    daily_ema50_aligned = align_htf_to_ltf(prices, df_1d, daily_ema50)
    daily_vol_ma_aligned = align_htf_to_ltf(prices, df_1d, daily_vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(daily_ema50_aligned[i]) or 
            np.isnan(daily_vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Current values
        dh = donchian_high[i]
        dl = donchian_low[i]
        dm = donchian_mid[i]
        daily_ema = daily_ema50_aligned[i]
        daily_vol_ma = daily_vol_ma_aligned[i]
        
        if position == 0:
            # LONG: Price breaks above Donchian upper band with price > daily EMA50 and volume > 1.5x daily MA
            if close[i] > dh and close[i] > daily_ema and volume[i] > daily_vol_ma * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower band with price < daily EMA50 and volume > 1.5x daily MA
            elif close[i] < dl and close[i] < daily_ema and volume[i] > daily_vol_ma * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Donchian midpoint OR below daily EMA50
            if close[i] < dm or close[i] < daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Donchian midpoint OR above daily EMA50
            if close[i] > dm or close[i] > daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals