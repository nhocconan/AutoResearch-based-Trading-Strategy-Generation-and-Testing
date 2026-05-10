#!/usr/bin/env python3
"""
1h_Trend_Follower_v1
Hypothesis: Use 4h RSI for momentum direction and 1d EMA200 for long-term trend, with 1h price action for entry timing.
Aims for 15-30 trades/year by requiring alignment of momentum and trend filters.
Works in bull/bear via trend filter + avoids counter-trend entries.
"""

name = "1h_Trend_Follower_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for RSI momentum
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate 4h RSI(14)
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Get 1d data for EMA200 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 1h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_1h = pd.Series(atr_1h).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 4h RSI (14), 1d EMA200 (200), 1h ATR MA (50)
    start_idx = max(14, 200, 50)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(atr_ma_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher timeframe filters
        rsi = rsi_4h_aligned[i]
        ema_200 = ema_200_1d_aligned[i]
        atr = atr_ma_1h[i]
        
        # Momentum: 4h RSI > 55 for long, < 45 for short
        mom_long = rsi > 55
        mom_short = rsi < 45
        
        # Trend: price above/below 1d EMA200
        trend_long = close[i] > ema_200
        trend_short = close[i] < ema_200
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr > 0  # Always true, but keeps structure for potential adjustment
        
        if position == 0:
            # Long entry: bullish momentum + uptrend
            if mom_long and trend_long and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short entry: bearish momentum + downtrend
            elif mom_short and trend_short and vol_filter:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: momentum turns bearish or trend breaks
            if not mom_long or not trend_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: momentum turns bullish or trend breaks
            if not mom_short or not trend_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals