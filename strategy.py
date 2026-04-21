#!/usr/bin/env python3
"""
1h_4h_1d_Trend_Follow_With_Pullback_Entry_V1
Hypothesis: In strong trends (4h EMA20 > EMA50), buy pullbacks to 1h VWAP with volume confirmation; sell short when trend reverses. Uses 1d ADX to filter weak markets. Designed for 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # --- 4h Trend Filter ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # --- 1d ADX Filter (trend strength) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    # Smooth TR, +DM, -DM
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # --- 1h VWAP ---
    typical_price = (prices['high'] + prices['low'] + prices['close']) / 3
    vwap_num = (typical_price * prices['volume']).cumsum()
    vwap_den = prices['volume'].cumsum()
    vwap = vwap_num / vwap_den
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        vwap_val = vwap.iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Trend condition: 4h EMA20 > EMA50 for long, < for short
        uptrend = ema20_4h_aligned[i] > ema50_4h_aligned[i]
        downtrend = ema20_4h_aligned[i] < ema50_4h_aligned[i]
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long: uptrend + strong trend + pullback to VWAP + volume
            if uptrend and strong_trend and (abs(price - vwap_val) < 0.005 * vwap_val) and volume_ok:
                signals[i] = 0.20
                position = 1
            # Short: downtrend + strong trend + pullback to VWAP + volume
            elif downtrend and strong_trend and (abs(price - vwap_val) < 0.005 * vwap_val) and volume_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal or ADX weak
            if not (uptrend and strong_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: trend reversal or ADX weak
            if not (downtrend and strong_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_1d_Trend_Follow_With_Pullback_Entry_V1"
timeframe = "1h"
leverage = 1.0