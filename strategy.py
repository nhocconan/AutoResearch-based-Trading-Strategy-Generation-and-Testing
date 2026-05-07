#!/usr/bin/env python3
name = "12h_Triple_Screen_Bollinger_KAMA_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    # 1d Bollinger Bands for trend filter
    close_1d = df_1d['close'].values
    bb_period = 20
    bb_std = 2.0
    sma_1d = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_1d = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb_1d = sma_1d + bb_std * std_1d
    lower_bb_1d = sma_1d - bb_std * std_1d
    
    # 1d trend: price above upper BB = uptrend, below lower BB = downtrend
    trend_up_1d = close_1d > upper_bb_1d
    trend_down_1d = close_1d < lower_bb_1d
    
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    trend_down_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_down_1d)
    
    # 12h KAMA for momentum
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.zeros_like(change)
    for i in range(len(change)):
        if np.sum(volatility[max(0, i-9):i+1]) > 0:
            er[i] = change[i] / np.sum(volatility[max(0, i-9):i+1])
        else:
            er[i] = 0
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 12h price relative to KAMA
    price_above_kama = close > kama
    price_below_kama = close < kama
    
    # 12h Bollinger Bands for volatility breakout
    bb_period_12h = 20
    bb_std_12h = 2.0
    sma_12h = pd.Series(close).rolling(window=bb_period_12h, min_periods=bb_period_12h).mean().values
    std_12h = pd.Series(close).rolling(window=bb_period_12h, min_periods=bb_period_12h).std().values
    upper_bb_12h = sma_12h + bb_std_12h * std_12h
    lower_bb_12h = sma_12h - bb_std_12h * std_12h
    
    # 12h volume spike: > 2.0x 20-period average
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume > 2.0 * vol_ma_12h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 34)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_up_1d_aligned[i]) or np.isnan(trend_down_1d_aligned[i]) or 
            np.isnan(kama[i]) or np.isnan(sma_12h[i]) or np.isnan(std_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1d uptrend + price above KAMA + break above upper BB + volume spike
            if (trend_up_1d_aligned[i] and price_above_kama[i] and 
                close[i] > upper_bb_12h[i] and vol_spike_12h[i]):
                signals[i] = 0.25
                position = 1
            # Short: 1d downtrend + price below KAMA + break below lower BB + volume spike
            elif (trend_down_1d_aligned[i] and price_below_kama[i] and 
                  close[i] < lower_bb_12h[i] and vol_spike_12h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below KAMA or volatility collapse (BB width < 50% of 20-period avg)
            bb_width_12h = upper_bb_12h[i] - lower_bb_12h[i]
            avg_bb_width = np.mean(bb_width_12h[max(0, i-19):i+1]) if i >= 20 else bb_width_12h
            if (close[i] < kama[i] or bb_width_12h < 0.5 * avg_bb_width):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above KAMA or volatility collapse
            bb_width_12h = upper_bb_12h[i] - lower_bb_12h[i]
            avg_bb_width = np.mean(bb_width_12h[max(0, i-19):i+1]) if i >= 20 else bb_width_12h
            if (close[i] > kama[i] or bb_width_12h < 0.5 * avg_bb_width):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals