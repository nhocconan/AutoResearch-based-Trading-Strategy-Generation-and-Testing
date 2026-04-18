#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume Spike and Weekly EMA Trend Filter
Hypothesis: Donchian(20) breakouts on 12h timeframe capture medium-term trends.
Volume confirmation ensures breakout validity, while weekly EMA50 filters for trend direction.
Designed for 12-37 trades/year on 12h timeframe with strict entry criteria to minimize fee drag.
Works in both bull and bear markets by filtering trades with weekly trend and requiring volume confirmation.
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
    
    # Get weekly data for EMA filter (once before loop)
    df_w = get_htf_data(prices, '1w')
    
    # Weekly EMA50 for trend filter
    ema_50 = pd.Series(df_w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_wk_aligned = align_htf_to_ltf(prices, df_w, ema_50)
    
    # Donchian channels (20-period) on 12h data
    # Highest high of last 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lowest low of last 20 periods
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # Need enough data for Donchian and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(ema_wk_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema_wk = ema_wk_aligned[i]
        
        if position == 0:
            # Long: break above upper Donchian with volume spike and price above weekly EMA50 (uptrend)
            if price > upper and volume_spike[i] and price > ema_wk:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume spike and price below weekly EMA50 (downtrend)
            elif price < lower and volume_spike[i] and price < ema_wk:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to lower Donchian or weekly EMA50
            if price < lower or price < ema_wk:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to upper Donchian or weekly EMA50
            if price > upper or price > ema_wk:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian_Breakout_VolumeSpike_WeeklyEMA50"
timeframe = "12h"
leverage = 1.0