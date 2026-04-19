#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1w EMA50 for trend and 1d KAMA for momentum, with volume confirmation.
# Uses 1d volume spike (>2x 20-period average) and 12h session filter (08-20 UTC) to reduce noise.
# Targets 15-30 trades/year (60-120 total over 4 years) with strict entry conditions.
# Trend-following with momentum confirmation works in bull/bear by following higher timeframe trends.
name = "12h_1wEMA50_1dKAMA_Volume"
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
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for EMA50 trend (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for KAMA momentum (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # KAMA parameters
    period = 10
    fast_ema = 2
    slow_ema = 30
    change = np.abs(np.diff(close_1d, period))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[period] = close_1d[period]
    for i in range(period+1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Volume filter: volume > 2.0 * 20-period average (1d volume)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 1w EMA50 AND above 1d KAMA with volume spike
            if (close[i] > ema_50_1w_aligned[i] and 
                close[i] > kama_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below 1w EMA50 AND below 1d KAMA with volume spike
            elif (close[i] < ema_50_1w_aligned[i] and 
                  close[i] < kama_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 1w EMA50 or 1d KAMA
            if close[i] < ema_50_1w_aligned[i] or close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above 1w EMA50 or 1d KAMA
            if close[i] > ema_50_1w_aligned[i] or close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals