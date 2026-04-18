#!/usr/bin/env python3
"""
1d Donchian Breakout with Weekly EMA Trend and Volume Spike
Hypothesis: Daily Donchian(20) breakouts with volume confirmation and weekly EMA34 trend filter
capture strong momentum moves. Works in both bull and bear markets by filtering trades with
the weekly trend and requiring volume confirmation to avoid false breakouts.
Designed for 10-30 trades/year on 1d timeframe. Low trade frequency minimizes fee drag.
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
    
    # Get weekly data for EMA (once before loop)
    df_w = get_htf_data(prices, '1w')
    
    # Weekly EMA34 for trend filter
    ema_34 = pd.Series(df_w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_w_aligned = align_htf_to_ltf(prices, df_w, ema_34)
    
    # Daily Donchian channels (20-period)
    # Using rolling window on daily data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike: 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # Need enough data for Donchian and EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_w_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema = ema_w_aligned[i]
        
        if position == 0:
            # Long: break above upper Donchian with volume spike and price above weekly EMA (uptrend)
            if price > upper and volume_spike[i] and price > ema:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume spike and price below weekly EMA (downtrend)
            elif price < lower and volume_spike[i] and price < ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to lower Donchian or weekly EMA
            if price < lower or price < ema:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to upper Donchian or weekly EMA
            if price > upper or price > ema:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_WeeklyEMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0