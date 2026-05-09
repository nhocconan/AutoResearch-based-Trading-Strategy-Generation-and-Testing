#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyKAMA_Trend_RSI_Filter"
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
    
    # Get weekly data for KAMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on weekly close
    close_1w = df_1w['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1w, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1w, n=1)), axis=0)  # 10-period volatility
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility[9:]])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.full_like(close_1w, np.nan)
    kama[29] = close_1w[29]  # start after 30 periods
    for i in range(30, len(close_1w)):
        if np.isnan(kama[i-1]):
            kama[i] = close_1w[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Weekly RSI for filter
    rsi_period = 14
    delta = np.diff(close_1w)
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    # First average
    avg_up = np.mean(up[:rsi_period]) if len(up) >= rsi_period else np.nan
    avg_down = np.mean(down[:rsi_period]) if len(down) >= rsi_period else np.nan
    rs = np.full_like(close_1w, np.nan)
    rsi = np.full_like(close_1w, np.nan)
    if not np.isnan(avg_down) and avg_down != 0:
        rs[rsi_period] = avg_up / avg_down
        rsi[rsi_period] = 100 - (100 / (1 + rs[rsi_period]))
    # Wilder smoothing
    for i in range(rsi_period + 1, len(close_1w)):
        avg_up = (up[i-1] + (rsi_period - 1) * avg_up) / rsi_period if not np.isnan(avg_up) else np.nan
        avg_down = (down[i-1] + (rsi_period - 1) * avg_down) / rsi_period if not np.isnan(avg_down) else np.nan
        if not np.isnan(avg_down) and avg_down != 0:
            rs[i] = avg_up / avg_down
            rsi[i] = 100 - (100 / (1 + rs[i]))
    
    # Align KAMA and RSI to daily
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    # Daily volume filter: volume > 1.5 * 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 20)  # Need enough data for KAMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: price above KAMA, RSI not overbought, volume confirmation
            if close[i] > kama_val and rsi_val < 70 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA, RSI not oversold, volume confirmation
            elif close[i] < kama_val and rsi_val > 30 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI overbought
            if close[i] < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI oversold
            if close[i] > kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals