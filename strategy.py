#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Divergence
Hypothesis: KAMA adapts to market efficiency, filtering noise in ranging markets while capturing trends. Combined with RSI divergence and volume confirmation, it works in both bull and bear markets by avoiding false signals during low-efficiency periods. Targets 15-25 trades/year by requiring KAMA trend alignment, RSI divergence, and volume > 1.5x 20-period average.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly KAMA for trend filter
    close_1w = df_1w['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1w, n=10))
    volatility = np.sum(np.abs(np.diff(close_1w, n=1)), axis=0)
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # KAMA calculation
    kama_1w = np.full_like(close_1w, np.nan)
    kama_1w[9] = close_1w[9]  # Start after 10 periods
    for i in range(10, len(close_1w)):
        if np.isnan(kama_1w[i-1]):
            kama_1w[i] = close_1w[i]
        else:
            kama_1w[i] = kama_1w[i-1] + sc[i] * (close_1w[i] - kama_1w[i-1])
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Daily RSI for divergence
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: >1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1w_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly KAMA
        uptrend = close[i] > kama_1w_aligned[i]
        downtrend = close[i] < kama_1w_aligned[i]
        
        # RSI divergence (simplified: look for RSI extremums)
        rsi_high = rsi[i] == np.max(rsi[max(0, i-5):i+1]) and rsi[i] > 70
        rsi_low = rsi[i] == np.min(rsi[max(0, i-5):i+1]) and rsi[i] < 30
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry logic: trend alignment with RSI extreme and volume
        long_entry = vol_confirm and uptrend and rsi_low
        short_entry = vol_confirm and downtrend and rsi_high
        
        # Exit logic: opposite RSI extreme or trend change
        long_exit = rsi_high or (not uptrend)
        short_exit = rsi_low or (not downtrend)
        
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

name = "1d_KAMA_Trend_With_RSI_Divergence"
timeframe = "1d"
leverage = 1.0