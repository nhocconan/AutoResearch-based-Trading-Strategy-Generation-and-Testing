#!/usr/bin/env python3
"""
4h_Donchian20_Trend_VolumeBreakout
Hypothesis: Trade breakouts of 20-period Donchian channels on 4h with volume confirmation and EMA34 trend filter. Long when price breaks above upper band with volume > 1.5x average and EMA34 rising; short when breaks below lower band with volume > 1.5x average and EMA34 falling. Uses 1d EMA34 for trend filter to avoid counter-trend trades. Targets 25-40 trades/year via strict breakout conditions. Works in bull/bear by following trend filter. Volume breakout filters low-momentum false breakouts.
"""

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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for trend filter
    ema_period = 34
    ema_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 / (ema_period + 1)) + (ema_1d[i-1] * (ema_period - 1) / (ema_period + 1))
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h Donchian(20) channels
    dc_period = 20
    upper = np.full_like(high, np.nan)
    lower = np.full_like(low, np.nan)
    if len(high) >= dc_period:
        for i in range(dc_period-1, len(high)):
            upper[i] = np.max(high[i-dc_period+1:i+1])
            lower[i] = np.min(low[i-dc_period+1:i+1])
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_period = 20
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(dc_period, vol_period, ema_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: break above upper band + volume + EMA34 rising (trend up)
            if close[i] > upper[i] and vol_confirm and ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band + volume + EMA34 falling (trend down)
            elif close[i] < lower[i] and vol_confirm and ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA34 or breaks below lower band
            if close[i] < ema_1d_aligned[i] or close[i] < lower[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA34 or breaks above upper band
            if close[i] > ema_1d_aligned[i] or close[i] > upper[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Trend_VolumeBreakout"
timeframe = "4h"
leverage = 1.0