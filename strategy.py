#!/usr/bin/env python3
name = "12h_Donchian_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Load 1D data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1D EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1D volume moving average for volume confirmation
    volume_1d_series = pd.Series(volume_1d)
    vol_ma_20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12H Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align 1D indicators to 12H timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for Donchian and indicators
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1D EMA50
        price_above_ema = close[i] > ema50_1d_aligned[i]
        price_below_ema = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period 1D volume MA
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # LONG: Donchian breakout above upper band + uptrend + volume
            if close[i] > donchian_high[i] and price_above_ema and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # SHORT: Donchian breakdown below lower band + downtrend + volume
            elif close[i] < donchian_low[i] and price_below_ema and volume_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Donchian lower band or trend reverses
            if close[i] < donchian_low[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Donchian upper band or trend reverses
            if close[i] > donchian_high[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals