#!/usr/bin/env python3
"""
6h_Stochastic_Divergence_1wTrend
Hypothesis: The 6h stochastic oscillator (K% crossing D%) provides early momentum signals, while the 1-week trend (price > SMA50) filters for higher-probability trades. Divergence between price and stochastic adds confluence for reversals in both bull and bear markets. Volume confirmation ensures conviction. Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag on 6h timeframe.
"""

name = "6h_Stochastic_Divergence_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1w data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1w Trend Filter: SMA50 ---
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # --- 6h Stochastic Oscillator (14,3,3) ---
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    # Avoid division by zero
    denom = highest_high - lowest_low
    denom = np.where(denom == 0, 1, denom)
    k_percent = 100 * ((close - lowest_low) / denom)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    # --- 6h RSI(14) for divergence detection ---
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # --- Volume Filter: spike above 1.5x median of last 20 periods ---
    vol_median = pd.Series(volume).rolling(window=20, min_periods=10).median().values
    vol_threshold = vol_median * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60  # for stochastic and SMA50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(sma50_1w_aligned[i]) or np.isnan(k_percent[i]) or 
            np.isnan(d_percent[i]) or np.isnan(rsi[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                # Hold position until exit signal
                signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1w trend
        trend_up = close[i] > sma50_1w_aligned[i]
        trend_down = close[i] < sma50_1w_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_threshold[i]
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        bull_div = False
        if i >= 2:
            bull_div = (low[i] < low[i-1] and low[i-1] < low[i-2]) and \
                       (rsi[i] > rsi[i-1] and rsi[i-1] > rsi[i-2])
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        bear_div = False
        if i >= 2:
            bear_div = (high[i] > high[i-1] and high[i-1] > high[i-2]) and \
                       (rsi[i] < rsi[i-1] and rsi[i-1] < rsi[i-2])
        
        if position == 0:
            # Long: stochastic bullish crossover + uptrend + volume + bullish divergence
            if (k_percent[i] > d_percent[i] and k_percent[i-1] <= d_percent[i-1] and
                trend_up and vol_ok and bull_div):
                signals[i] = 0.25
                position = 1
            # Short: stochastic bearish crossover + downtrend + volume + bearish divergence
            elif (k_percent[i] < d_percent[i] and k_percent[i-1] >= d_percent[i-1] and
                  trend_down and vol_ok and bear_div):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: stochastic bearish crossover OR trend reversal
                if (k_percent[i] < d_percent[i] and k_percent[i-1] >= d_percent[i-1]) or \
                   not trend_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: stochastic bullish crossover OR trend reversal
                if (k_percent[i] > d_percent[i] and k_percent[i-1] <= d_percent[i-1]) or \
                   not trend_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals