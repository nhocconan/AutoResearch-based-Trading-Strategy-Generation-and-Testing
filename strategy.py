#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend filter + 1d RSI mean reversion + volume spike
# Uses Kaufman's Adaptive Moving Average (KAMA) on 12h for trend direction to avoid whipsaws
# 1d RSI(14) for mean reversion entries: long when RSI<30, short when RSI>70
# Volume confirmation (2.0x 20-bar MA) ensures institutional participation
# Designed for 50-150 total trades over 4 years (12-37/year) on 12h timeframe
# Works in bull markets (trend continuation) and bear markets (extreme mean reversion)
# BTC and ETH focused with SOL as validation

name = "12h_KAMA_Trend_1dRSI_MeanRev_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h KAMA for trend filter (ER=10, fast=2, slow=30)
    def kama(close, er_period=10, fast=2, slow=30):
        close_s = pd.Series(close)
        change = np.abs(close_s.diff(er_period)).values
        volatility = np.abs(close_s.diff()).rolling(window=er_period, min_periods=1).sum().values
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_12h = kama(close, er_period=10, fast=2, slow=30)
    
    # Calculate 1d RSI(14) for mean reversion
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: 2.0x 20-period average (20*12h = 10 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for KAMA and RSI)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(kama_12h[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price above KAMA (bullish trend) AND RSI < 30 (oversold) AND volume spike
            if (close[i] > kama_12h[i] and 
                rsi_1d_aligned[i] < 30 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Price below KAMA (bearish trend) AND RSI > 70 (overbought) AND volume spike
            elif (close[i] < kama_12h[i] and 
                  rsi_1d_aligned[i] > 70 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price below KAMA (trend change) OR RSI > 50 (mean reversion exit)
            if close[i] < kama_12h[i] or rsi_1d_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price above KAMA (trend change) OR RSI < 50 (mean reversion exit)
            if close[i] > kama_12h[i] or rsi_1d_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals