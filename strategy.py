#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation
    # Donchian(20) provides clear structure for breakouts in both bull/bear markets
    # 12h EMA(50) ensures alignment with higher timeframe trend to avoid counter-trend trades
    # Volume > 2x 20-period average confirms institutional participation
    # Target: 30-50 trades/year per symbol to minimize fee drag and avoid overtrading
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50) for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period) on 4h
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_20[i] = np.max(high[i-20:i])
        lowest_20[i] = np.min(low[i-20:i])
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    vol_ratio = np.full(n, np.nan)
    for i in range(20, n):
        if vol_ma_20[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma_20[i]
        else:
            vol_ratio[i] = 1.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 12h EMA(50)
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Breakout conditions
        breakout_long = (close[i] > highest_20[i]) and (vol_ratio[i] > 2.0) and uptrend
        breakout_short = (close[i] < lowest_20[i]) and (vol_ratio[i] > 2.0) and downtrend
        
        # Exit conditions: return to midpoint of Donchian channel
        midpoint_20 = (highest_20[i] + lowest_20[i]) / 2.0
        long_exit = close[i] < midpoint_20
        short_exit = close[i] > midpoint_20
        
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.30
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -0.30
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_donchian_breakout_vol_trend_v1"
timeframe = "4h"
leverage = 1.0