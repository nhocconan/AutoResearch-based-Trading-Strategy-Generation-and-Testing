#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_RSI_Momentum_Volume
Hypothesis: Uses daily KAMA trend direction with RSI momentum (30/70) and volume confirmation (1.5x 20-day average) to capture medium-term moves. Designed for low trade frequency (7-25/year) to minimize fee drift while capturing sustained trends. Works in both bull and bear by following KAMA trend direction. Targets 30-100 total trades over 4 years.
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
    
    # Get weekly data for trend filter (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly KAMA for trend filter
    close_1w = df_1w['close'].values
    # Calculate Efficiency Ratio for KAMA
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.abs(np.diff(close_1w))
    er = np.zeros_like(close_1w)
    er[1:] = change[1:] / (np.abs(np.diff(close_1w)) + 1e-10)
    er_cumsum = np.cumsum(er)
    er_volatility = np.cumsum(np.abs(np.diff(close_1w)))
    er[1:] = change[1:] / (er_volatility[1:] + 1e-10)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Align weekly KAMA to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # Daily RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily volume confirmation: >1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly KAMA
        uptrend = close[i] > kama_aligned[i]
        downtrend = close[i] < kama_aligned[i]
        
        # Momentum filter: RSI in momentum zone
        rsi_momentum_long = rsi[i] > 50
        rsi_momentum_short = rsi[i] < 50
        
        # Volume confirmation (>1.5x average)
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry conditions
        long_entry = uptrend and rsi_momentum_long and vol_confirm
        short_entry = downtrend and rsi_momentum_short and vol_confirm
        
        # Exit conditions: trend reversal
        long_exit = not uptrend
        short_exit = not downtrend
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_KAMA_Trend_Filter_RSI_Momentum_Volume"
timeframe = "1d"
leverage = 1.0