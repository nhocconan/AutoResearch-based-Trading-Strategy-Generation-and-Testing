#!/usr/bin/env python3
# 1h_adx_ema_pullback_4h1d_v1
# Hypothesis: 1h EMA(21) pullback to 4h EMA(50) with ADX(14) trend filter and 1d trend confirmation.
# In trending markets (ADX > 25), price pulls back to the 4h EMA(50) before continuing.
# 1d EMA(50) ensures alignment with daily trend, reducing counter-trend trades.
# Designed for 15-30 trades/year (60-120 over 4 years) with tight entry conditions.
# Works in bull/bear markets: trend following with pullback entries improves win rate.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_adx_ema_pullback_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for EMA50 and ADX
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA(50)
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 4h ADX(14)
    plus_dm = np.diff(high_4h, prepend=high_4h[0])
    minus_dm = np.diff(low_4h, prepend=low_4h[0]) * -1
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr = np.maximum(
        high_4h[1:] - low_4h[1:],
        np.maximum(
            np.abs(high_4h[1:] - close_4h[:-1]),
            np.abs(low_4h[1:] - close_4h[:-1])
        )
    )
    tr = np.concatenate([[high_4h[0] - low_4h[0]], tr])
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Get 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h EMA(21) for entry timing
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_21[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 1h EMA(21) or ADX weakens
            if close[i] < ema_21[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above 1h EMA(21) or ADX weakens
            if close[i] > ema_21[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: price pulls back to 4h EMA(50) from above, ADX strong, above 1d EMA(50)
            if (low[i] <= ema_50_4h_aligned[i] <= high[i]) and \
               (close[i] > ema_50_4h_aligned[i]) and \
               (adx_aligned[i] > 25) and \
               (close[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Enter short: price pulls back to 4h EMA(50) from below, ADX strong, below 1d EMA(50)
            elif (low[i] <= ema_50_4h_aligned[i] <= high[i]) and \
                 (close[i] < ema_50_4h_aligned[i]) and \
                 (adx_aligned[i] > 25) and \
                 (close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals