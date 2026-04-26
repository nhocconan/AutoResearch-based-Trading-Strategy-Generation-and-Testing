#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrendFilter_VolumeSpike_v1
Hypothesis: 1d Donchian(20) breakout with 1w trend filter (price > 1w EMA50 for long, < for short) and volume confirmation (>1.5x 20-day avg volume).
Enters long when price breaks above Donchian(20) upper band AND close > 1w EMA50 AND volume > 1.5x 20-day avg volume.
Enters short when price breaks below Donchian(20) lower band AND close < 1w EMA50 AND volume > 1.5x 20-day avg volume.
Exits on opposite Donchian breakout or when price re-enters the Donchian channel.
Donchian breakouts capture strong momentum moves; 1w EMA50 filter ensures alignment with higher timeframe trend to avoid counter-trend trades.
Volume confirmation reduces false breakouts. Targets 7-25 trades/year (30-100 total over 4 years) on 1d timeframe.
Works in bull/bear markets by trading with the 1w trend and using Donchian channels as objective breakout levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Donchian channels (20-period)
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-day average volume
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 20 for volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or 
            np.isnan(vol_ma20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > period20_high[i-1]  # break above previous upper band
        breakout_down = close[i] < period20_low[i-1]  # break below previous lower band
        reentry = (close[i] >= period20_low[i] and close[i] <= period20_high[i])  # price re-enters channel
        
        if position == 0:
            # Long: breakout up AND close > 1w EMA50 AND volume spike
            if breakout_up and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout down AND close < 1w EMA50 AND volume spike
            elif breakout_down and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: breakout down OR price re-enters channel
            if breakout_down or reentry:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: breakout up OR price re-enters channel
            if breakout_up or reentry:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrendFilter_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0