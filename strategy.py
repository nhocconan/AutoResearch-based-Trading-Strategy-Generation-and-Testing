#!/usr/bin/env python3
"""
4h_Donchian_20_Volume_Trend
Hypothesis: Breakouts beyond Donchian(20) channels on 4h timeframe capture strong momentum.
Volume confirmation ensures breakouts are genuine, while 1d EMA34 trend filter aligns with institutional bias.
Works in bull/bear by following breakout direction with trend filter. Target: 20-35 trades/year (80-140 total).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Donchian channels (20-period) - using 4h data directly
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # 1-day EMA trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_4h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0
    bars_since_entry = 0
    
    start_idx = 20  # Warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_filter[i]) or np.isnan(ema_1d_4h[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_ok = volume_filter[i]
        ema_trend = ema_1d_4h[i]
        
        if position == 0:
            # Long: break above upper Donchian with volume in uptrend
            if price > upper and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: break below lower Donchian with volume in downtrend
            elif price < lower and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            bars_since_entry += 1
            # Minimum holding period: 3 bars
            if bars_since_entry < 3:
                signals[i] = 0.25
            else:
                signals[i] = 0.25
                # Exit: price returns to lower Donchian or trend reverses
                if price < lower or price < ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        
        elif position == -1:
            bars_since_entry += 1
            # Minimum holding period: 3 bars
            if bars_since_entry < 3:
                signals[i] = -0.25
            else:
                signals[i] = -0.25
                # Exit: price returns to upper Donchian or trend reverses
                if price > upper or price > ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "4h_Donchian_20_Volume_Trend"
timeframe = "4h"
leverage = 1.0