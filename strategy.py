#!/usr/bin/env python3
name = "6h_True_Range_Channel_With_1d_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # 1d True Range (TR) for volatility-based channel
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range components
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.inf  # First value has no previous close
    tr3[0] = np.inf
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(10) using Wilder's smoothing
    atr_1d = np.zeros_like(tr_1d)
    atr_1d[9] = np.mean(tr_1d[:10])  # Initial SMA
    for i in range(10, len(tr_1d)):
        atr_1d[i] = (atr_1d[i-1] * 9 + tr_1d[i]) / 10
    
    # Upper and Lower Channels (ATR-based)
    upper_channel_1d = close_1d + 1.5 * atr_1d
    lower_channel_1d = close_1d - 1.5 * atr_1d
    
    # Align channels to 6h timeframe
    upper_channel_aligned = align_htf_to_ltf(prices, df_1d, upper_channel_1d)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1d, lower_channel_1d)
    
    # 1d EMA20 for trend filter
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # 6m volume filter: > 1.3x 20-period average (~5 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.3 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Wait for volume MA and EMA20
    
    for i in range(start_idx, n):
        if (np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or 
            np.isnan(ema20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price touches lower channel AND above EMA20 (bullish trend) with volume
            if low[i] <= lower_channel_aligned[i] and close[i] > ema20_1d_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price touches upper channel AND below EMA20 (bearish trend) with volume
            elif high[i] >= upper_channel_aligned[i] and close[i] < ema20_1d_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price crosses above EMA20 OR touches upper channel
            if close[i] > ema20_1d_aligned[i] or high[i] >= upper_channel_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price crosses below EMA20 OR touches lower channel
            if close[i] < ema20_1d_aligned[i] or low[i] <= lower_channel_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s mean reversion within a 1d ATR-based channel, filtered by 1d EMA20 trend.
# Long when price touches/touches lower 1.5*ATR channel while above EMA20 (bullish trend) with volume confirmation.
# Short when price touches/touches upper channel while below EMA20 (bearish trend) with volume confirmation.
# Exits when price crosses EMA20 or touches opposite channel.
# Uses 1d timeframe for channel calculation and trend to avoid whipsaws, 6s for entry timing.
# Volume filter (>1.3x average) ensures conviction. Discrete 0.25 position size limits risk.
# Works in trending markets (trend filter) and ranging markets (channel mean reversion).
# Target: 15-35 trades/year to minimize fee drain while capturing meaningful moves.