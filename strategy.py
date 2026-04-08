#!/usr/bin/env python3
# 12h_1d_1w_trend_follow_volume_v1
# Hypothesis: 12-hour price channel breakout with daily trend filter and weekly momentum filter captures sustained moves while avoiding chop. Uses 12h Donchian breakout, daily EMA trend, and weekly RSI momentum. Designed for 12-37 trades/year with low frequency to minimize fee drag in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_trend_follow_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for momentum filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 12h Donchian(20) for breakout signals
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate weekly RSI(14) for momentum filter
    close_1w = df_1w['close'].values
    delta = pd.Series(close_1w).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi14_1w = 100 - (100 / (1 + rs))
    rsi14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi14_1w.values)
    
    # Volume confirmation: volume > 1.3x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.3
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi14_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 12h Donchian low OR weekly RSI < 40 (loss of momentum)
            if close[i] < donchian_low[i] or rsi14_1w_aligned[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above 12h Donchian high OR weekly RSI > 60 (loss of momentum)
            if close[i] > donchian_high[i] or rsi14_1w_aligned[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above 12h Donchian high, above daily EMA50, weekly RSI > 50, with volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema50_1d_aligned[i] and rsi14_1w_aligned[i] > 50 and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 12h Donchian low, below daily EMA50, weekly RSI < 50, with volume confirmation
            elif close[i] < donchian_low[i] and close[i] < ema50_1d_aligned[i] and rsi14_1w_aligned[i] < 50 and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals