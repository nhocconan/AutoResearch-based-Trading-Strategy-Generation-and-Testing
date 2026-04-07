#!/usr/bin/env python3
"""
12h ADX Trend with Volume and ATR Filter
Long when ADX > 25 and +DI > -DI with volume confirmation
Short when ADX > 25 and -DI > +DI with volume confirmation
Exit when ADX < 20 or ADX crossover reverses
Uses 1d trend filter to avoid counter-trend trades
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_adx_trend_volume_filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === ADX Calculation ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])  # Simple average for first value
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
        return result
    
    period = 14
    tr_sum = wilder_smooth(tr, period)
    plus_dm_sum = wilder_smooth(plus_dm, period)
    minus_dm_sum = wilder_smooth(minus_dm, period)
    
    # DI and ADX
    plus_di = 100 * plus_dm_sum / (tr_sum + 1e-10)
    minus_di = 100 * minus_dm_sum / (tr_sum + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = wilder_smooth(dx, period)
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === 1d Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    # 50-period EMA on 1d
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        if np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or np.isnan(vol_ratio[i]) or np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of 1d EMA
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: ADX weak (< 20) or DI crossover
            if adx[i] < 20 or minus_di[i] > plus_di[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ADX weak (< 20) or DI crossover
            if adx[i] < 20 or plus_di[i] > minus_di[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Strong trend condition
            if adx[i] > 25:
                # Long: +DI > -DI and bullish trend
                if plus_di[i] > minus_di[i] and bullish_trend:
                    position = 1
                    signals[i] = 0.25
                # Short: -DI > +DI and bearish trend
                elif minus_di[i] > plus_di[i] and bearish_trend:
                    position = -1
                    signals[i] = -0.25
    
    return signals