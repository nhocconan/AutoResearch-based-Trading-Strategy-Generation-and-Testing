#!/usr/bin/env python3
"""
12h_1w_KAMA_Direction_RSI_Overbought
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a trend filter that avoids whipsaws in choppy markets. Combined with RSI overbought/oversold levels and weekly trend confirmation, this strategy aims to capture sustained moves while minimizing false signals. Designed for low trade frequency on 12h timeframe to reduce fee drag, with weekly trend filter ensuring trades align with higher-timeframe momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly KAMA for trend direction
    close_weekly = df_weekly['close'].values
    # Efficiency Ratio
    change = np.abs(close_weekly - np.roll(close_weekly, 10))
    change[0:10] = np.abs(close_weekly[0:10] - close_weekly[0])
    vol = np.sum(np.abs(np.diff(close_weekly, prepend=close_weekly[0])), axis=0) if len(close_weekly) > 1 else 0
    er = np.where(vol != 0, change / vol, 0)
    # Smoothing constants
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    # KAMA calculation
    kama_weekly = np.zeros_like(close_weekly)
    kama_weekly[0] = close_weekly[0]
    for i in range(1, len(close_weekly)):
        kama_weekly[i] = kama_weekly[i-1] + sc[i] * (close_weekly[i] - kama_weekly[i-1])
    # Align weekly KAMA to 12h
    kama_weekly_aligned = align_htf_to_ltf(prices, df_weekly, kama_weekly)
    
    # Load daily data for RSI
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily RSI(14)
    close_daily = df_daily['close'].values
    delta = np.diff(close_daily, prepend=close_daily[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_daily = 100 - (100 / (1 + rs))
    # Align daily RSI to 12h
    rsi_daily_aligned = align_htf_to_ltf(prices, df_daily, rsi_daily)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h KAMA for entry signal
    change_12h = np.abs(close - np.roll(close, 10))
    change_12h[0:10] = np.abs(close[0:10] - close[0])
    vol_12h = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if len(close) > 1 else 0
    er_12h = np.where(vol_12h != 0, change_12h / vol_12h, 0)
    sc_12h = (er_12h * (0.6667 - 0.0645) + 0.0645) ** 2
    kama_12h = np.zeros_like(close)
    kama_12h[0] = close[0]
    for i in range(1, len(close)):
        kama_12h[i] = kama_12h[i-1] + sc_12h[i] * (close[i] - kama_12h[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(kama_weekly_aligned[i]) or np.isnan(rsi_daily_aligned[i]) or np.isnan(kama_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        weekly_kama = kama_weekly_aligned[i]
        daily_rsi = rsi_daily_aligned[i]
        kama_12h_val = kama_12h[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_ma = np.mean(volume[max(0, i-20):i]) if i >= 20 else volume[i]
        vol_ok = vol_current > 1.5 * vol_ma
        
        if position == 0:
            # Long: price above 12h KAMA, weekly KAMA uptrend, RSI not overbought, volume confirmation
            if price > kama_12h_val and weekly_kama > kama_weekly_aligned[i-1] and daily_rsi < 70 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below 12h KAMA, weekly KAMA downtrend, RSI not oversold, volume confirmation
            elif price < kama_12h_val and weekly_kama < kama_weekly_aligned[i-1] and daily_rsi > 30 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 12h KAMA or weekly trend turns down
            if price < kama_12h_val or weekly_kama < kama_weekly_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 12h KAMA or weekly trend turns up
            if price > kama_12h_val or weekly_kama > kama_weekly_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_KAMA_Direction_RSI_Overbought"
timeframe = "12h"
leverage = 1.0