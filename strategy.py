#!/usr/bin/env python3

# 1h_4h1d_multi_factor_v1
# Hypothesis: Multi-factor strategy using 4h trend (EMA50) and 1d momentum (RSI14) for direction, 
# with 1h RSI pullback entries. Filters by session (08-20 UTC) to reduce noise.
# Designed to work in both bull and bear markets by combining trend following with mean reversion entries.
# Target: 15-35 trades/year for low fee drag on 1h timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_multi_factor_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h trend filter (EMA50) - load once before loop
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d momentum filter (RSI14) - load once before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate RSI14 on daily data
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi14_1d = 100 - (100 / (1 + rs))
    rsi14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi14_1d)
    
    # 1h RSI for entry timing
    delta_h = np.diff(close, prepend=close[0])
    gain_h = np.where(delta_h > 0, delta_h, 0)
    loss_h = np.where(delta_h < 0, -delta_h, 0)
    avg_gain_h = pd.Series(gain_h).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss_h = pd.Series(loss_h).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs_h = avg_gain_h / (avg_loss_h + 1e-10)
    rsi14_h = 100 - (100 / (1 + rs_h))
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi14_1d_aligned[i]) or np.isnan(rsi14_h[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Determine trend direction from 4h EMA50 and 1d RSI
        trend_up = close[i] > ema50_4h_aligned[i] and rsi14_1d_aligned[i] > 50
        trend_down = close[i] < ema50_4h_aligned[i] and rsi14_1d_aligned[i] < 50
        
        if position == 1:  # Long position
            # Exit: trend reversal or overbought RSI
            if not trend_up or rsi14_h[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: trend reversal or oversold RSI
            if not trend_down or rsi14_h[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry during session
            if in_session:
                # Long entry: pullback in uptrend
                if trend_up and rsi14_h[i] < 40 and rsi14_h[i] > rsi14_h[i-1]:
                    position = 1
                    signals[i] = 0.20
                # Short entry: pullback in downtrend
                elif trend_down and rsi14_h[i] > 60 and rsi14_h[i] < rsi14_h[i-1]:
                    position = -1
                    signals[i] = -0.20
    
    return signals