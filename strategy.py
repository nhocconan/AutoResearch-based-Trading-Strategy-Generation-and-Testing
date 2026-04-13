#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d KAMA trend with 1w RSI regime filter and volume confirmation
    # Long: KAMA rising AND 1w RSI > 50 (bullish bias) AND volume > 1.3x avg
    # Short: KAMA falling AND 1w RSI < 50 (bearish bias) AND volume > 1.3x avg
    # Exit: opposite KAMA direction or volume dry-up
    # Using 1d timeframe for low trade frequency, KAMA for adaptive trend,
    # 1w RSI for regime filter (avoid counter-trend trades), volume for confirmation.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for RSI regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly RSI(14) for regime filter
    close_1w = df_1w['close'].values
    
    # RSI calculation
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: smoothed = (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    avg_gain = wilders_smoothing(gain, 14)
    avg_loss = wilders_smoothing(loss, 14)
    
    # Avoid division by zero
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to 1d
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate daily KAMA(10, 2, 30)
    # Efficiency Ratio (ER) = |net change| / sum(|changes|)
    # Smoothing Constant (SC) = [ER * (fastest - slowest) + slowest]^2
    # KAMA = previous KAMA + SC * (price - previous KAMA)
    
    # Parameters: fast=2, slow=30, lookback=10
    fast_sc = 2 / (2 + 1)  # 0.6667
    slow_sc = 2 / (30 + 1)  # 0.0645
    lookback = 10
    
    # Calculate ER
    net_change = np.abs(np.diff(close, prepend=close[0]))
    total_change = np.zeros(n)
    for i in range(1, n):
        total_change[i] = total_change[i-1] + np.abs(close[i] - close[i-1])
        if i >= lookback:
            total_change[i] -= np.abs(close[i-lookback] - close[i-lookback+1])
    
    # Avoid division by zero
    er = np.where(total_change > 0, net_change / total_change, 0)
    
    # Calculate SC
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = close[i]
    
    # KAMA direction (1 = rising, -1 = falling, 0 = flat)
    kama_dir = np.zeros(n)
    for i in range(1, n):
        if kama[i] > kama[i-1]:
            kama_dir[i] = 1
        elif kama[i] < kama[i-1]:
            kama_dir[i] = -1
        else:
            kama_dir[i] = kama_dir[i-1]
    
    # Get daily volume for confirmation (>1.3x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(kama_dir[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: RSI > 50 = bullish bias, RSI < 50 = bearish bias
        bullish_bias = rsi_1w_aligned[i] > 50
        bearish_bias = rsi_1w_aligned[i] < 50
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: KAMA direction + regime bias + volume confirmation
        long_entry = (kama_dir[i] == 1) and bullish_bias and vol_confirm
        short_entry = (kama_dir[i] == -1) and bearish_bias and vol_confirm
        
        # Exit logic: opposite KAMA direction or volume dry-up
        long_exit = (kama_dir[i] == -1) or not vol_confirm
        short_exit = (kama_dir[i] == 1) or not vol_confirm
        
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
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_kama_rsi_volume_v1"
timeframe = "1d"
leverage = 1.0