#!/usr/bin/env python3
"""
1d Donchian(20) Breakout with Weekly EMA Trend and Volume Spike
Hypothesis: Daily Donchian breakouts capture major trends. Weekly EMA filter ensures
trading only in the direction of the weekly trend. Volume spikes confirm breakout
strength. Designed for 30-100 trades over 4 years (7-25/year) with low turnover
to minimize fee impact. Works in both bull and bear markets via trend filtering.
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
    ema_34_w = pd.Series(df_w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_w_aligned = align_htf_to_ltf(prices, df_w, ema_34_w)
    
    # Donchian channels (20-period) on daily
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ATR for stop loss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 40  # enough for Donchian and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(ema_34_w_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        weekly_ema = ema_34_w_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: break above upper Donchian with volume spike and price above weekly EMA (uptrend)
            if price > upper and volume_spike[i] and price > weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume spike and price below weekly EMA (downtrend)
            elif price < lower and volume_spike[i] and price < weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to lower Donchian or ATR trailing stop
            if price <= lower or price < (high[i] - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to upper Donchian or ATR trailing stop
            if price >= upper or price > (low[i] + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_WeeklyEMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0