#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout + 12h EMA34 trend + volume confirmation.
- Primary timeframe: 4h for execution, HTF: 12h for EMA trend filter.
- Camarilla levels from 1d: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2.
- Long when price > H3 and 12h EMA34 slope up; Short when price < L3 and 12h EMA34 slope down.
- Volume confirmation: current volume > 2.0 * 20-period volume MA to ensure strong participation.
- Regime filter: only trade when 12h ADX(14) > 20 to avoid choppy markets.
- Discrete signal size: 0.30 to balance profit potential and drawdown control.
- Target: 100-200 total trades over 4 years (25-50/year) for 4h timeframe.
- Works in bull via buying strength in uptrend, in bear via selling weakness in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA34 and ADX for trend/regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # EMA34 for trend direction
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # ADX(14) for regime filter
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_di_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    plus_di = 100 * plus_di_smooth / atr
    minus_di = 100 * minus_di_smooth / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate 1d Camarilla levels (H3, L3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    camarilla_h3_1d = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 2
    camarilla_l3_1d = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 2
    camarilla_h3_1d = camarilla_h3_1d.values
    camarilla_l3_1d = camarilla_l3_1d.values
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 buffer + volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when ADX > 20 (avoid choppy markets)
        if adx_aligned[i] > 20:
            if position == 0:
                # EMA34 slope for trend confirmation
                if i > 0 and not np.isnan(ema_34_12h_aligned[i]) and not np.isnan(ema_34_12h_aligned[i-1]):
                    ema34_slope = ema_34_12h_aligned[i] - ema_34_12h_aligned[i-1]
                    
                    # Long: price > H3 and EMA34 slope up
                    if close[i] > h3_aligned[i] and ema34_slope > 0 and volume_spike[i]:
                        signals[i] = 0.30
                        position = 1
                    # Short: price < L3 and EMA34 slope down
                    elif close[i] < l3_aligned[i] and ema34_slope < 0 and volume_spike[i]:
                        signals[i] = -0.30
                        position = -1
            elif position == 1:
                # Long exit: price returns to midpoint of H3/L3 or breaks below L3
                midpoint = (h3_aligned[i] + l3_aligned[i]) / 2
                if close[i] < midpoint or close[i] < l3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            elif position == -1:
                # Short exit: price returns to midpoint or breaks above H3
                midpoint = (h3_aligned[i] + l3_aligned[i]) / 2
                if close[i] > midpoint or close[i] > h3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
        else:
            # In choppy market (ADX <= 20), flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_H3L3_12hEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0