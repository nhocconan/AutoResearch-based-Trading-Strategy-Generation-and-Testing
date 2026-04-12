#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_Pullback_Volume
Hypothesis: In 1h timeframe, buy pullbacks to 4h EMA20 during 1d uptrend with volume confirmation, sell rallies to 4h EMA20 during 1d downtrend.
Uses 1d trend filter to avoid counter-trend trades, 4h EMA20 as dynamic support/resistance, and volume spike for entry confirmation.
Session filter (08-20 UTC) reduces noise. Low trade frequency (~20-40/year) minimizes fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Camarilla_Pullback_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4H EMA20 FOR DYNAMIC SUPPORT/RESISTANCE ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # === 1D TREND FILTER (EMA50) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === VOLUME CONFIRMATION (1H) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # === SESSION FILTER: 08-20 UTC ===
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(hours[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Long: price pulls back to 4h EMA20 during 1d uptrend with volume spike
        long_signal = (close[i] >= ema20_4h_aligned[i] * 0.998) and \
                      (close[i] <= ema20_4h_aligned[i] * 1.002) and \
                      (close[i] > ema50_1d_aligned[i]) and \
                      (vol_ratio[i] > 1.8)
        
        # Short: price rallies to 4h EMA20 during 1d downtrend with volume spike
        short_signal = (close[i] <= ema20_4h_aligned[i] * 1.002) and \
                       (close[i] >= ema20_4h_aligned[i] * 0.998) and \
                       (close[i] < ema50_1d_aligned[i]) and \
                       (vol_ratio[i] > 1.8)
        
        # Exit when price moves 0.6% away from EMA20 (avoid whipsaw)
        exit_long = close[i] > ema20_4h_aligned[i] * 1.006 and position == 1
        exit_short = close[i] < ema20_4h_aligned[i] * 0.994 and position == -1
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals