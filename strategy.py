#!/usr/bin/env python3
"""
1h Donchian Breakout + Volume Spike + ADX Trend Filter
Hypothesis: 1h timeframe captures intermediate-term trends. Donchian breakouts identify momentum, volume spikes confirm institutional participation, and ADX > 25 filters for trending markets. Using 4h/1d for trend direction reduces whipsaw in ranging markets while 1h provides precise entry timing. Designed for 15-35 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend direction and ADX
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for longer-term trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = np.full_like(high_4h, np.nan)
    donchian_low = np.full_like(low_4h, np.nan)
    
    for i in range(20, len(high_4h)):
        donchian_high[i] = np.max(high_4h[i-20:i])
        donchian_low[i] = np.min(low_4h[i-20:i])
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate 4h ADX for trend strength
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1)) if 'close_4h' in locals() else np.abs(high_4h - np.roll(df_4h['close'].values, 1))
    tr3 = np.abs(low_4h - np.roll(df_4h['close'].values, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Use close from df_4h for TR calculation
    close_4h = df_4h['close'].values
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    up_move = high_4h - np.roll(high_4h, 1)
    down_move = np.roll(low_4h, 1) - low_4h
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    def smooth_series(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_4h = smooth_series(tr, 14)
    plus_di = 100 * smooth_series(plus_dm, 14) / np.where(atr_4h != 0, atr_4h, 1)
    minus_di = 100 * smooth_series(minus_dm, 14) / np.where(atr_4h != 0, atr_4h, 1)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_4h = smooth_series(dx, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Calculate 1d EMA for long-term trend filter
    close_1d = df_1d['close'].values
    ema_1d = np.zeros_like(close_1d)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 49) / 51
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1])
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(adx_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Enter long: price breaks above 4h Donchian high + ADX > 25 + volume spike + price > 1d EMA
            if (close[i] > donchian_high_aligned[i] and 
                adx_4h_aligned[i] > 25 and 
                vol_spike[i] and 
                close[i] > ema_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below 4h Donchian low + ADX > 25 + volume spike + price < 1d EMA
            elif (close[i] < donchian_low_aligned[i] and 
                  adx_4h_aligned[i] > 25 and 
                  vol_spike[i] and 
                  close[i] < ema_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 4h Donchian low or ADX weakens
            if close[i] < donchian_low_aligned[i] or adx_4h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price breaks above 4h Donchian high or ADX weakens
            if close[i] > donchian_high_aligned[i] or adx_4h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian_Breakout_VolumeSpike_ADXFilter"
timeframe = "1h"
leverage = 1.0