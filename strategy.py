#!/usr/bin/env python3
"""
6h_1d_ADX_Regime_Trend
Hypothesis: On 6h timeframe, ADX-based trend regime filtering (ADX>25 = trend, ADX<20 = range) 
combined with EMA crossovers provides robust signals in both bull and bear markets.
ADX regime filter prevents whipsaws in sideways markets while capturing trends.
Target: 15-30 trades/year per symbol.
"""

name = "6h_1d_ADX_Regime_Trend"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        def WilderSmooth(data, period):
            result = np.zeros_like(data)
            alpha = 1.0 / period
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
            return result
        
        tr_smooth = WilderSmooth(tr, period)
        plus_dm_smooth = WilderSmooth(plus_dm, period)
        minus_dm_smooth = WilderSmooth(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # DX and ADX
        dx = np.zeros_like(close)
        dx_mask = (plus_di + minus_di) > 0
        dx[dx_mask] = 100 * np.abs(plus_di[dx_mask] - minus_di[dx_mask]) / (plus_di[dx_mask] + minus_di[dx_mask])
        
        adx = WilderSmooth(dx, period)
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d trend: 34 EMA
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = close_1d > ema_34_1d
    downtrend_1d = close_1d < ema_34_1d
    
    # Align 1d trend to 6h
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Calculate 6m EMA crossovers (8 and 21)
    ema_8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Golden cross (bullish) and death cross (bearish)
    golden_cross = ema_8 > ema_21
    death_cross = ema_8 < ema_21
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get aligned values
        adx_val = adx[i]
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        gc = golden_cross[i]
        dc = death_cross[i]
        
        if position == 0:
            # LONG: ADX>25 (trending) + 1d uptrend + golden cross
            if adx_val > 25 and uptrend and gc:
                signals[i] = 0.25
                position = 1
            # SHORT: ADX>25 (trending) + 1d downtrend + death cross
            elif adx_val > 25 and downtrend and dc:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: ADX<20 (ranging) or death cross
            if adx_val < 20 or dc:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: ADX<20 (ranging) or golden cross
            if adx_val < 20 or gc:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals