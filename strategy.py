#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Price breaking above Camarilla R1 or below S1 on 12h with 1d trend and volume confirmation captures institutional breakouts. 
Works in bull/bear by following 1d EMA34 trend. Low trade frequency avoids fee drag.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Daily trend: EMA34
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    uptrend_1d = close > ema_34_1d_aligned
    downtrend_1d = close < ema_34_1d_aligned
    
    # Camarilla levels from previous day
    typical_price = (high + low + close) / 3.0
    df_1d_tp = pd.Series(typical_price).rolling(window=1, min_periods=1).mean().values  # just tp
    df_1d_for_pivots = get_htf_data(pd.DataFrame({
        'high': df_1d['high'].values,
        'low': df_1d['low'].values,
        'close': df_1d['close'].values
    }), '1d')
    high_1d = df_1d_for_pivots['high'].values
    low_1d = df_1d_for_pivots['low'].values
    close_1d = df_1d_for_pivots['close'].values
    
    # Calculate Camarilla for each day
    camarilla_R1 = np.full_like(close, np.nan)
    camarilla_S1 = np.full_like(close, np.nan)
    
    for i in range(len(df_1d)):
        if i == 0:
            continue
        H = high_1d[i-1]
        L = low_1d[i-1]
        C = close_1d[i-1]
        R1 = C + (H - L) * 1.1 / 12
        S1 = C - (H - L) * 1.1 / 12
        # Find indices in 12h data that belong to this day
        # We'll align later, for now store daily values
        camarilla_R1[i*2] = R1  # approximate: 2x 12h bars per day
        camarilla_S1[i*2] = S1
    
    # Better approach: calculate daily then align
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    R1_1d = typical_price_1d + (high_1d - low_1d) * 1.1 / 12
    S1_1d = typical_price_1d - (high_1d - low_1d) * 1.1 / 12
    R1_12h = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # Volume confirmation: > 1.5x 24-period average (2 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    for i in range(24, n):  # need 2 days of volume data
        if np.isnan(R1_12h[i]) or np.isnan(S1_12h[i]):
            signals[i] = 0.0
            continue
            
        # LONG: price > R1, uptrend, volume
        if close[i] > R1_12h[i] and uptrend_1d[i] and volume_confirm[i]:
            signals[i] = 0.25
        # SHORT: price < S1, downtrend, volume
        elif close[i] < S1_12h[i] and downtrend_1d[i] and volume_confirm[i]:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals