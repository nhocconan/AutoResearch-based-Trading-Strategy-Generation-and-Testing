#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_volume_v1"
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
    
    # Get weekly data for trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # KAMA on weekly data to determine trend direction
    close_1w = df_1w['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.abs(np.diff(close_1w))
    er = np.zeros_like(close_1w)
    er[1:] = change[1:] / (np.cumsum(volatility) + 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Align KAMA to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # Weekly RSI for momentum filter
    delta_1w = np.diff(close_1w, prepend=close_1w[0])
    gain_1w = np.where(delta_1w > 0, delta_1w, 0)
    loss_1w = np.where(delta_1w < 0, -delta_1w, 0)
    avg_gain_1w = pd.Series(gain_1w).rolling(window=14, min_periods=14).mean().values
    avg_loss_1w = pd.Series(loss_1w).rolling(window=14, min_periods=14).mean().values
    rs_1w = avg_gain_1w / (avg_loss_1w + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs_1w))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Daily volume filter - 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    # Daily RSI for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or
            np.isnan(volume_ok[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price above weekly KAMA, weekly RSI > 50, daily RSI < 30 (oversold), volume confirmation
        long_signal = (close[i] > kama_aligned[i] and 
                      rsi_1w_aligned[i] > 50 and 
                      rsi[i] < 30 and 
                      volume_ok[i])
        # Short: price below weekly KAMA, weekly RSI < 50, daily RSI > 70 (overbought), volume confirmation
        short_signal = (close[i] < kama_aligned[i] and 
                       rsi_1w_aligned[i] < 50 and 
                       rsi[i] > 70 and 
                       volume_ok[i])
        
        # Exit when daily RSI returns to neutral zone
        exit_long = rsi[i] > 50
        exit_short = rsi[i] < 50
        
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
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals