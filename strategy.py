#!/usr/bin/env python3
"""
1d_1w_AntiTrend_Reversion_v1
Hypothesis: Mean reversion at weekly extremes in 1d timeframe using Bollinger Bands and RSI.
Works in both bull and bear markets by fading extreme moves when price touches Bollinger Bands
with RSI confirmation, avoiding trend-following whipsaw. Target: 10-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_AntiTrend_Reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Bollinger Bands (20, 2.0)
    close_1w = df_1w['close'].values
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2.0 * std_20
    lower_bb = sma_20 - 2.0 * std_20
    
    # Align Bollinger Bands to daily
    upper_bb_daily = align_htf_to_ltf(prices, df_1w, upper_bb)
    lower_bb_daily = align_htf_to_ltf(prices, df_1w, lower_bb)
    
    # Calculate daily RSI (14) for confirmation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: avoid low-volume false signals
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_bb_daily[i]) or np.isnan(lower_bb_daily[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: require above-average volume
        volume_ok = volume[i] > vol_ma[i] * 0.8
        
        # Entry conditions: touch Bollinger Band with RSI extreme
        touch_upper = close[i] >= upper_bb_daily[i]
        touch_lower = close[i] <= lower_bb_daily[i]
        rsi_overbought = rsi[i] > 70
        rsi_oversold = rsi[i] < 30
        
        long_entry = touch_lower and rsi_oversold and volume_ok
        short_entry = touch_upper and rsi_overbought and volume_ok
        
        # Exit conditions: RSI returns to neutral range or opposite band touch
        rsi_neutral = (rsi[i] >= 30) and (rsi[i] <= 70)
        long_exit = rsi_neutral or touch_upper
        short_exit = rsi_neutral or touch_lower
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals