#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_WeeklyTrend_VolumeSpike_v1
Hypothesis: Daily Donchian(20) breakout with weekly EMA34 trend filter and volume spike captures strong directional moves in both bull and bear markets. Low trade frequency (~15-25/year) minimizes fee drag while allowing significant profit per trade. Weekly trend filter ensures we only trade in the direction of the higher timeframe momentum, reducing false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # === Daily Donchian(20) channels ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 20-period highest high and lowest low (using prior 20 periods only)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === Weekly EMA34 trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Volume spike filter (20-period on daily) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        upper_channel = highest_20[i]
        lower_channel = lowest_20[i]
        weekly_trend = ema_34_1w_aligned[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume spike > 2.0 + price above weekly EMA
            if price_close > upper_channel and vol_spike > 2.0 and price_close > weekly_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + volume spike > 2.0 + price below weekly EMA
            elif price_close < lower_channel and vol_spike > 2.0 and price_close < weekly_trend:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit logic: reverse signal or volatility-based stop
            if position == 1:
                # Exit long if price breaks below lower Donchian (opposite breakout)
                if price_close < lower_channel:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short if price breaks above upper Donchian (opposite breakout)
                if price_close > upper_channel:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0