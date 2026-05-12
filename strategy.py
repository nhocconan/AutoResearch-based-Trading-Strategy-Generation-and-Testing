#!/usr/bin/env python3
name = "6h_RSI_Trend_Squeeze"
timeframe = "6h"
leverage = 1.0

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
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # RSI(14) with momentum filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands width filter (squeeze detection)
    ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_width = (2 * 2 * std) / ma  # 2 std dev bands
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze = bb_width < bb_width_ma  # Bollinger squeeze condition
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(squeeze[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI > 50 (bullish momentum) + above 1d EMA34 + Bollinger squeeze
            if rsi[i] > 50 and close[i] > ema_34_1d_aligned[i] and squeeze[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI < 50 (bearish momentum) + below 1d EMA34 + Bollinger squeeze
            elif rsi[i] < 50 and close[i] < ema_34_1d_aligned[i] and squeeze[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI < 40 or below 1d EMA34
            if rsi[i] < 40 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI > 60 or above 1d EMA34
            if rsi[i] > 60 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals