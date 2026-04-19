#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h CRSI with 1d trend filter and volume confirmation
# CRSI (Connors RSI) captures short-term mean reversion extremes
# 1d EMA filter ensures trades align with higher timeframe trend
# Volume filter confirms institutional participation
# Designed to work in both bull and bear markets by trading pullbacks in trending markets
# Target: 20-40 trades/year per symbol (~80-160 total over 4 years)

name = "4h_CRSI_TrendVolume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate RSI components for CRSI
    # RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/3, adjust=False, min_periods=3).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/3, adjust=False, min_periods=3).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi3 = 100 - (100 / (1 + rs))
    
    # RSI of streak length
    up_days = np.zeros_like(close)
    down_days = np.zeros_like(close)
    for i in range(1, n):
        if close[i] > close[i-1]:
            up_days[i] = up_days[i-1] + 1
            down_days[i] = 0
        elif close[i] < close[i-1]:
            down_days[i] = down_days[i-1] + 1
            up_days[i] = 0
        else:
            up_days[i] = 0
            down_days[i] = 0
    
    # RSI(2) on streak
    up_change = np.where(up_days > 0, 1, 0)
    down_change = np.where(down_days > 0, 1, 0)
    avg_up = pd.Series(up_change).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_down = pd.Series(down_change).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs_streak = avg_up / (avg_down + 1e-10)
    rsi_streak = 100 - (100 / (1 + rs_streak))
    
    # Percent Rank(100) - percentage of values below current in lookback window
    def percentile_rank(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                result[i] = np.nan
            else:
                window_data = arr[i-window+1:i+1]
                rank = np.sum(window_data <= arr[i]) / window * 100
                result[i] = rank
        return result
    
    percent_rank = percentile_rank(close, 100)
    
    # CRSI = (RSI(3) + RSI(Streak) + PercentRank(100)) / 3
    crsi = (rsi3 + rsi_streak + percent_rank) / 3
    
    # 4h ATR for position sizing and stops
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: current volume > 1.3x average volume over last 20 periods
    avg_volume = np.full_like(volume, np.nan)
    for i in range(n):
        if i >= 20:
            avg_volume[i] = np.mean(volume[i-20:i])
        else:
            avg_volume[i] = volume[i] if i > 0 else 0
    volume_filter = volume > 1.3 * avg_volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34)  # Ensure CRSI and EMA are valid
    
    for i in range(start_idx, n):
        if np.isnan(crsi[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(atr_4h[i]) or np.isnan(volume_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: CRSI oversold (<15) + price above 1d EMA + volume confirmation
            if crsi[i] < 15 and close[i] > ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: CRSI overbought (>85) + price below 1d EMA + volume confirmation
            elif crsi[i] > 85 and close[i] < ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: CRSI crosses above 50 (mean reversion complete) or ATR stop
            if crsi[i] > 50 or close[i] < close[i-1] - 1.5 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: CRSI crosses below 50 (mean reversion complete) or ATR stop
            if crsi[i] < 50 or close[i] > close[i-1] + 1.5 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals