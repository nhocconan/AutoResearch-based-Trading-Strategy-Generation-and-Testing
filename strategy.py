#!/usr/bin/env python3
"""
6h Donchian Breakout + Daily Trend Filter + Volume Spike
Hypothesis: In BTC/ETH, strong trends persist across multiple days. A breakout above
the 20-period Donchian channel on 6h timeframe, when aligned with the daily trend
(price above/below daily EMA50) and confirmed by volume spikes, captures
trend continuation moves while avoiding counter-trend noise. Works in bull
markets (upside breakouts) and bear markets (downside breakouts). Targets 15-30
trades per year.
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
    
    # Get daily data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA50 for trend filter
    daily_close = df_1d['close'].values
    ema_50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h Donchian channel (20 periods)
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(50, lookback)  # need enough history
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema_50 = ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high + above daily EMA50 + volume spike
            if price > upper and price > ema_50 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + below daily EMA50 + volume spike
            elif price < lower and price < ema_50 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Hold long position
            signals[i] = 0.25
            # Exit: price breaks below Donchian low or below daily EMA50
            if price < lower or price < ema_50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Hold short position
            signals[i] = -0.25
            # Exit: price breaks above Donchian high or above daily EMA50
            if price > upper or price > ema_50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_DailyEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0