#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(15) breakout with 12-hour ADX(14) trend strength filter and volume confirmation.
# The 12-hour ADX(14) ensures trades occur only in trending markets (ADX > 25), avoiding whipsaws in ranges.
# Donchian(15) breakout captures momentum in the direction of the 12-hour trend.
# Volume > 1.3x the 20-period average confirms institutional participation.
# Exit when price crosses the 12-hour EMA(20) or breaks the opposite Donchian band.
# Designed for 15-30 trades per year per symbol (60-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for trend filter and EMA
    df_12h = get_htf_data(prices, '12h')
    
    # 12h ADX(14) for trend strength
    adx_len = 14
    if len(df_12h) < adx_len * 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = np.diff(high_12h)
    down_move = -np.diff(low_12h)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=adx_len, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=adx_len, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=adx_len, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=adx_len, adjust=False).mean().values
    adx_12h = np.concatenate([np.full(adx_len, np.nan), adx[adx_len:]])
    
    # 12h EMA(20) for exit
    ema_len = 20
    ema_12h = pd.Series(close_12h).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    
    # Align to 4h
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Donchian channel (15 periods) on 4h
    dc_len = 15
    dc_upper = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().shift(1).values
    dc_lower = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().shift(1).values
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, dc_len, 20, adx_len * 2)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or
            np.isnan(adx_12h_aligned[i]) or
            np.isnan(ema_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        trending = adx_12h_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: Donchian breakout above + trending + volume
            if (close[i] > dc_upper[i] and 
                trending and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Donchian breakdown below + trending + volume
            elif (close[i] < dc_lower[i] and 
                  trending and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 12h EMA20 or breaks below Donchian lower
            if close[i] < ema_12h_aligned[i] or close[i] < dc_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 12h EMA20 or breaks above Donchian upper
            if close[i] > ema_12h_aligned[i] or close[i] > dc_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_ADX14_Donchian_Volume_v1"
timeframe = "4h"
leverage = 1.0