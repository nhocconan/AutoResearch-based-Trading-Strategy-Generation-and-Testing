#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_HTF
Hypothesis: Donchian(20) breakout on 4h timeframe with 12h EMA34 trend filter and volume confirmation.
In bull markets, price breaks above upper band; in bear markets, breaks below lower band.
The 12h EMA34 filters trend direction to avoid counter-trend trades.
Volume confirmation ensures breakout strength.
Designed to work in both bull and bear regimes by following the higher timeframe trend.
Target: 20-40 trades/year to minimize fee drag while capturing significant moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h EMA34 for trend filter (loaded once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Warmup for Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_20[i]
        lower = low_20[i]
        trend = ema_34_12h_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band with volume in uptrend
            if price > upper and vol_conf and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band with volume in downtrend
            elif price < lower and vol_conf and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price closes below middle of Donchian channel or trend reverses
            mid = (upper + lower) / 2
            if price < mid or price < trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price closes above middle of Donchian channel or trend reverses
            mid = (upper + lower) / 2
            if price > mid or price > trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend_HTF"
timeframe = "4h"
leverage = 1.0