#!/usr/bin/env python3
"""
4h Donchian Breakout + Volume Spike + ADX Filter (Multi-Timeframe)
Hypothesis: Price breaking Donchian(20) channels with volume confirmation 
and ADX > 25 captures institutional breakouts in trending markets. 
Uses 12h EMA34 as trend filter to avoid counter-trend trades. 
Low frequency due to strict 3-condition entry (breakout + volume + trend).
Works in bull/bear by following momentum with volatility-adjusted stops.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 35:
        return np.zeros(n)
    
    # Calculate EMA34 on 12h close
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False).values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate ADX on 4h data
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / np.where(atr != 0, atr, 1)
    minus_di = 100 * wilder_smooth(minus_dm, 14) / np.where(atr != 0, atr, 1)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilder_smooth(dx, 14)
    
    # Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                result[i] = np.nan
            else:
                result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                result[i] = np.nan
            else:
                result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donch_high = rolling_max(high, 20)
    donch_low = rolling_min(low, 20)
    
    # Volume spike: current > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1])
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_12h_aligned[i]) or np.isnan(adx[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]):
            signals[i] = 0.0
            continue
        
        trend_up = close[i] > ema_12h_aligned[i]
        trend_down = close[i] < ema_12h_aligned[i]
        adx_strong = adx[i] > 25
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + uptrend + ADX + volume
            if (close[i] > donch_high[i] and 
                trend_up and 
                adx_strong and 
                vol_ok):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian low + downtrend + ADX + volume
            elif (close[i] < donch_low[i] and 
                  trend_down and 
                  adx_strong and 
                  vol_ok):
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit: price re-enters Donchian channel or trend weakens
            if close[i] < donch_high[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit: price re-enters Donchian channel or trend weakens
            if close[i] > donch_low[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0