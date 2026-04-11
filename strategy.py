#!/usr/bin/env python3
"""
6h_1w_momentum_reversal_v1
Strategy: 6h momentum reversal with weekly trend filter and volume confirmation
Timeframe: 6h
Leverage: 1.0
Hypothesis: Combines 6h RSI momentum extremes (oversold/overbought) with weekly trend direction to capture reversals in trending markets. Uses volume spike confirmation to filter false signals. Designed to work in both bull (buy oversold in uptrend) and bear (sell overbought in downtrend) markets by aligning with higher timeframe momentum. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_momentum_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 6h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly RSI(14) for trend filter
    close_1w = df_1w['close'].values
    delta_1w = np.diff(close_1w, prepend=close_1w[0])
    gain_1w = np.where(delta_1w > 0, delta_1w, 0)
    loss_1w = np.where(delta_1w < 0, -delta_1w, 0)
    avg_gain_1w = pd.Series(gain_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1w = pd.Series(loss_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1w = avg_gain_1w / (avg_loss_1w + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs_1w))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend: RSI > 50 = uptrend, < 50 = downtrend
        weekly_uptrend = rsi_1w_aligned[i] > 50
        weekly_downtrend = rsi_1w_aligned[i] < 50
        
        # 6h momentum extremes
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: oversold + volume spike in weekly uptrend
        long_signal = rsi_oversold and vol_confirmed and weekly_uptrend
        
        # Short: overbought + volume spike in weekly downtrend
        short_signal = rsi_overbought and vol_confirmed and weekly_downtrend
        
        # Exit when RSI returns to neutral zone (40-60)
        exit_long = position == 1 and rsi[i] > 40
        exit_short = position == -1 and rsi[i] < 60
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals