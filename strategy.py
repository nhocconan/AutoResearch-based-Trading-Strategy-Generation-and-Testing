#!/usr/bin/env python3
# 4h_1dATRBreakout_1wTrend
# Hypothesis: Breakout trades using 1d ATR-based channels (mean ± N*ATR) filtered by 1w trend.
# In bull markets: buy breakouts above upper channel when 1w trend is up.
# In bear markets: sell breakdowns below lower channel when 1w trend is down.
# Volatility-based channels adapt to market conditions, reducing false breakouts in low volatility.
# Trend filter ensures we trade with the dominant weekly direction.
# Designed for low frequency (~20-40 trades/year) to minimize fee drag and survive both bull and bear markets.

name = "4h_1dATRBreakout_1wTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d ATR (20-period) for volatility band calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # First period
    tr3[0] = np.abs(low_1d[0] - close_1d[0])   # First period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    
    # === 1d Close for center of ATR bands ===
    avg_price_1d = close_1d  # Using close as representative price
    avg_price_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_price_1d)
    
    # === ATR Bands: mean ± 1.5 * ATR ===
    upper_band = avg_price_1d_aligned + 1.5 * atr_20_aligned
    lower_band = avg_price_1d_aligned - 1.5 * atr_20_aligned
    
    # === 1w EMA40 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema_40_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1w EMA40
        trend_up = close[i] > ema_40_1w_aligned[i]
        trend_down = close[i] < ema_40_1w_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > upper_band[i]
        breakdown_down = close[i] < lower_band[i]
        
        if position == 0:
            # LONG: breakout above upper band with uptrend
            if breakout_up and trend_up:
                signals[i] = 0.25
                position = 1
            # SHORT: breakdown below lower band with downtrend
            elif breakdown_down and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: breakdown below lower band or trend reversal to down
            if breakdown_down or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: breakout above upper band or trend reversal to up
            if breakout_up or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals