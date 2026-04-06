#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend + RSI momentum + 1d volume filter
# Long when KAMA turns up + RSI > 50 + volume > 1.5x average
# Short when KAMA turns down + RSI < 50 + volume > 1.5x average
# Uses KAMA for adaptive trend following, RSI for momentum confirmation
# Volume filter ensures trades occur during active participation
# Target: 50-150 total trades over 4 years with controlled risk

name = "12h_kama_rsi_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # KAMA calculation (ER=10, fast=2, slow=30)
    def calculate_kama(close, er_period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=er_period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close)
    
    # RSI calculation (14-period)
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
        avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: KAMA turns down or RSI < 40
            if kama[i] < kama[i-1] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: KAMA turns up or RSI > 60
            if kama[i] > kama[i-1] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Long: KAMA turning up + RSI > 50 + volume spike
            if (kama[i] > kama[i-1] and 
                rsi[i] > 50 and
                volume[i] > 1.5 * volume_ma_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: KAMA turning down + RSI < 50 + volume spike
            elif (kama[i] < kama[i-1] and 
                  rsi[i] < 50 and
                  volume[i] > 1.5 * volume_ma_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals