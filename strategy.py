#!/usr/bin/env python3
"""
Hypothesis: 12-hour strategy using 1-day Williams %R for mean reversion and 1-week RSI for trend filter.
Long when 1-day Williams %R < -80 (oversold) and 1-week RSI < 50 (weak trend) with volume confirmation.
Short when 1-day Williams %R > -20 (overbought) and 1-week RSI > 50 (strong trend) with volume confirmation.
Exit when Williams %R returns to -50 or volatility expands.
Designed for low turnover: ~15-30 trades/year per symbol to minimize fee drift.
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
    
    # Load 1-day data once for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1-day Williams %R calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    williams_r = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        highest_high = np.max(high_1d[i-14:i+1])
        lowest_low = np.min(low_1d[i-14:i+1])
        if highest_high != lowest_low:
            williams_r[i] = -100 * (highest_high - close_1d[i]) / (highest_high - lowest_low)
        else:
            williams_r[i] = -50
    
    # Load 1-week data once for RSI
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1-week RSI calculation (14-period)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(20, n):
        # 1-day index (2 bars per day: 24/12 = 2)
        idx_1d = i // 2
        if idx_1d < 14:  # need enough for Williams %R
            continue
        
        # Get previous 1-day Williams %R to avoid look-ahead
        williams_prev = williams_r[idx_1d - 1] if idx_1d - 1 < len(williams_r) else williams_r[-1]
        if np.isnan(williams_prev):
            continue
        
        # Create arrays for alignment (using previous values)
        williams_arr = np.full(len(df_1d), williams_prev)
        williams_12h = align_htf_to_ltf(prices, df_1d, williams_arr)[i]
        
        # 1-week index (14 bars per week: 7*24/12 = 14)
        idx_1w = i // 14
        if idx_1w < 14:  # need enough for RSI
            continue
        
        # Get previous 1-week RSI to avoid look-ahead
        rsi_prev = rsi_1w[idx_1w - 1] if idx_1w - 1 < len(rsi_1w) else rsi_1w[-1]
        if np.isnan(rsi_prev):
            continue
        
        # Create arrays for alignment (using previous values)
        rsi_arr = np.full(len(df_1w), rsi_prev)
        rsi_12h = align_htf_to_ltf(prices, df_1w, rsi_arr)[i]
        
        if position == 0:
            # Long: Williams %R oversold + RSI weak trend + volume surge
            if (williams_12h < -80 and 
                rsi_12h < 50 and 
                volume[i] > vol_ma[i] * 1.5):
                position = 1
                signals[i] = position_size
            # Short: Williams %R overbought + RSI strong trend + volume surge
            elif (williams_12h > -20 and 
                  rsi_12h > 50 and 
                  volume[i] > vol_ma[i] * 1.5):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Williams %R returns to -50 or volatility expansion (simplified)
            if williams_12h > -50:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Williams %R returns to -50 or volatility expansion (simplified)
            if williams_12h < -50:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_1d_WilliamsR_1wRSI_Volume"
timeframe = "12h"
leverage = 1.0