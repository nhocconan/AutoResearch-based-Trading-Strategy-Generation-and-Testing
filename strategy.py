#!/usr/bin/env python3
"""
12h ATR Breakout with Daily Trend Filter
Long when price breaks above ATR-based upper band with daily uptrend
Short when price breaks below ATR-based lower band with daily downtrend
Exit when price crosses opposite band or trend reverses
Designed for low-frequency trading (12-37 trades/year) with volatility filtering
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_atr_breakout_daily_trend_v1"
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
    
    # === ATR Calculation (14-period) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === ATR Bands (2.5 * ATR) ===
    ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    upper_band = ma + 2.5 * atr
    lower_band = ma - 2.5 * atr
    
    # === Daily Trend Filter (using 1d data) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(atr[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below lower band OR daily trend turns down
            if close[i] < lower_band[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above upper band OR daily trend turns up
            if close[i] > upper_band[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volatility filter (ATR not too low)
            if atr[i] < 0.5 * np.nanmedian(atr[max(0, i-50):i]):
                signals[i] = 0.0
                continue
            
            # Entry: breakout with daily trend alignment
            if close[i] > upper_band[i] and close[i] > ema_50_1d_aligned[i]:
                # Break above upper band with daily uptrend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < lower_band[i] and close[i] < ema_50_1d_aligned[i]:
                # Break below lower band with daily downtrend -> short
                position = -1
                signals[i] = -0.25
    
    return signals