#!/usr/bin/env python3
"""
1d_1w_Donchian_Breakout_Trend
Hypothesis: Trade weekly Donchian(20) breakouts with daily RSI confirmation and volume filter. 
Breakouts above/below 20-day high/low capture trend continuation in both bull and bear markets.
Daily RSI > 55 for long, < 45 for short ensures momentum alignment. Volume > 1.5x average filters weak breakouts.
Designed for 10-20 trades/year with clear trend-following logic that works in bull (breakouts continue) and bear (breakouts fail, reverse) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Donchian_Breakout_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY DATA FOR DONCHIAN CHANNEL ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian(20) channels
    donch_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian to daily timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # === DAILY INDICATORS: RSI AND VOLUME ===
    # RSI(14)
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price breaks above weekly Donchian high with momentum and volume
        long_signal = (close[i] > donch_high_aligned[i] and 
                      rsi[i] > 55 and 
                      volume[i] > (vol_ma[i] * 1.5))
        
        # Short: price breaks below weekly Donchian low with momentum and volume
        short_signal = (close[i] < donch_low_aligned[i] and 
                       rsi[i] < 45 and 
                       volume[i] > (vol_ma[i] * 1.5))
        
        # Exit: price returns to opposite Donchian level or RSI reverses
        exit_long = (position == 1 and 
                    (close[i] < donch_low_aligned[i] or rsi[i] < 40))
        exit_short = (position == -1 and 
                     (close[i] > donch_high_aligned[i] or rsi[i] > 60))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals