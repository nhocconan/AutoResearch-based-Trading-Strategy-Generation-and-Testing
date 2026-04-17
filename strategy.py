# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 1d KAMA trend with 1w EMA50 filter, RSI momentum, and volume confirmation.
- KAMA adapts to market noise, reducing whipsaws in sideways markets.
- Weekly EMA50 provides strong trend filter to avoid counter-trend trades.
- RSI(14) > 50 for long, < 50 for short ensures momentum alignment.
- Volume > 1.5x 20-period average confirms conviction.
- Designed for fewer, higher-quality trades to minimize fee drag and improve generalization.
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
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate daily KAMA trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # KAMA components: ER (efficiency ratio) and smoothing constants
    change = np.abs(np.diff(close_1d, k=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_1d, k=1)), axis=0)  # 10-period volatility
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants: fastest = 2/(2+1), slowest = 2/(30+1)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Start after 10 periods
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate daily RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need weekly EMA50, KAMA, RSI, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filters
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        price_above_weekly_ema = close[i] > ema50_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema50_1w_aligned[i]
        
        # RSI momentum
        rsi_above_50 = rsi_aligned[i] > 50
        rsi_below_50 = rsi_aligned[i] < 50
        
        if position == 0:
            # Long: Price above KAMA and weekly EMA50, RSI > 50, with volume
            if (price_above_kama and price_above_weekly_ema and rsi_above_50 and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA and weekly EMA50, RSI < 50, with volume
            elif (price_below_kama and price_below_weekly_ema and rsi_below_50 and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below KAMA or weekly EMA50
            if (close[i] < kama_aligned[i]) or (close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above KAMA or weekly EMA50
            if (close[i] > kama_aligned[i]) or (close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_WeeklyEMA50_RSI_Volume"
timeframe = "1d"
leverage = 1.0