#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_Trend_PriceAction_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily KAMA for trend ---
    price_diff = np.abs(np.diff(close, prepend=close[0]))
    direction = np.abs(np.diff(close, k=10, prepend=close[:10]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # --- Weekly ADX for regime filter ---
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.concatenate([[np.nan], close_1w[:-1]]))
    tr3 = np.abs(low_1w - np.concatenate([[np.nan], close_1w[:-1]]))
    tr = np.maximum.reduce([tr1, tr2, tr3])
    
    # Directional Movement
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = np.diff(low_1w, prepend=low_1w[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def _wilder_smooth(arr, period):
        smoothed = np.full_like(arr, np.nan, dtype=float)
        smoothed[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    atr = _wilder_smooth(tr, 14)
    plus_di = 100 * _wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * _wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = _wilder_smooth(dx, 14)
    
    # Align weekly ADX to daily
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # --- Volume filter ---
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        adx_val = adx_aligned[i]
        
        # Volume filter: current volume > 1.8x 20-period average
        volume_ok = vol > 1.8 * vol_ma
        
        # Trend filter: ADX > 20 for trending market
        trend_ok = adx_val > 20
        
        if position == 0:
            # Long: price > KAMA AND volume + trend confirmation
            if price > kama[i] and volume_ok and trend_ok:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA AND volume + trend confirmation
            elif price < kama[i] and volume_ok and trend_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below KAMA
            if price < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above KAMA
            if price > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals