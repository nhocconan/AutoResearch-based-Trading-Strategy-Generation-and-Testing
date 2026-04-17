#!/usr/bin/env python3
"""
1d Keltner Channel Breakout + Volume Spike + ADX Trend Filter
Long: Close > Upper KC(20,2) + ADX > 25 + Volume > 1.5x 20-period average
Short: Close < Lower KC(20,2) + ADX > 25 + Volume > 1.5x 20-period average
Exit: Opposite signal or Close crosses middle KC line (EMA20)
Uses weekly trend filter to avoid counter-trend trades in strong trends.
Designed to work in both bull and bear markets by capturing volatility expansions.
Target: 30-100 total trades over 4 years (7-25/year)
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(20) for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Keltner Channel components (20,2)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(high - low).rolling(window=20, min_periods=20).mean().values
    upper_kc = ema_20 + 2 * atr
    lower_kc = ema_20 - 2 * atr
    middle_kc = ema_20  # EMA20 as middle line
    
    # Calculate volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    tr = np.maximum(high - low, np.absolute(high - np.roll(low, 1)), np.absolute(low - np.roll(high, 1)))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(30, 20)  # need sufficient data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(upper_kc[i]) or np.isnan(lower_kc[i]) or
            np.isnan(middle_kc[i]) or np.isnan(vol_avg[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_avg_val = vol_avg[i]
        adx_val = adx[i]
        upper = upper_kc[i]
        lower = lower_kc[i]
        middle = middle_kc[i]
        weekly_trend = ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: Close > Upper KC + ADX > 25 + Volume spike + Weekly uptrend (price > weekly EMA20)
            if price > upper and adx_val > 25 and vol > 1.5 * vol_avg_val and price > weekly_trend:
                signals[i] = 0.25
                position = 1
            # Short: Close < Lower KC + ADX > 25 + Volume spike + Weekly downtrend (price < weekly EMA20)
            elif price < lower and adx_val > 25 and vol > 1.5 * vol_avg_val and price < weekly_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close < Middle KC or weekly trend turns down
            if price < middle or price < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close > Middle KC or weekly trend turns up
            if price > middle or price > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Keltner_Breakout_Volume_ADX_WeeklyTrend"
timeframe = "1d"
leverage = 1.0