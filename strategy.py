#!/usr/bin/env python3
"""
4h_1d_donchian_volume_trend_v1
Strategy: 4h Donchian breakout with volume confirmation and 1d trend filter
Timeframe: 4h
Leverage: 1.0
Hypothesis: Buy breakouts above 20-period Donchian high with volume > 1.5x average volume in uptrend (price > 100 EMA on 1d). Sell breakdowns below Donchian low with volume > 1.5x average volume in downtrend (price < 100 EMA on 1d). Uses volume confirmation to avoid false breakouts and trend filter to trade with higher timeframe momentum. Designed to work in both bull and bear markets by following the 1d trend. Target: 20-50 trades per year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d EMA100 for trend filter
    close_1d = df_1d['close'].values
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_100_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        vol_current = volume[i]
        
        # Trend filters from 1d
        uptrend_1d = price_close > ema_100_1d_aligned[i]
        downtrend_1d = price_close < ema_100_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = vol_current > 1.5 * vol_ma[i]
        
        # Breakout conditions
        breakout_up = price_close > donchian_high[i]
        breakdown_down = price_close < donchian_low[i]
        
        # Long: Donchian breakout up with volume in uptrend
        long_signal = breakout_up and volume_confirm and uptrend_1d
        
        # Short: Donchian breakdown down with volume in downtrend
        short_signal = breakdown_down and volume_confirm and downtrend_1d
        
        # Exit when price returns to middle of Donchian channel
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