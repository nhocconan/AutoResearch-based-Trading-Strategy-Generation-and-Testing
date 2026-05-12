#!/usr/bin/env python3
# 6h Bollinger Band Width + RSI Mean Reversion + Weekly Trend
# Hypothesis: In ranging markets (low BB width), RSI extremes provide mean-reversion opportunities.
# In trending markets (high BB width), we avoid trades to prevent whipsaw.
# Weekly trend filter ensures we only take mean-reversion trades in the direction of the higher timeframe trend.
# Works in both bull and bear markets by adapting to volatility regimes and using weekly trend filter.
# Designed for low trade frequency (~15-30/year) with clear entry/exit rules.

name = "6h_BBW_RSI_MeanReversion_WeeklyTrend"
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
    
    # === Weekly Data for Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_close_1w = df_1w['close'].values
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(weekly_close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_6h = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === Bollinger Band Width (20, 2) on 6h ===
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # === RSI (14) on 6h ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure BB and RSI ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bb_width[i]) or np.isnan(rsi[i]) or np.isnan(ema_20_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Regime filter: only trade when BB width is low (rangy market)
        is_ranging = bb_width[i] < 0.05  # 5% width threshold
        
        if position == 0 and is_ranging:
            # LONG: RSI oversold (<30) and price above weekly EMA20 (uptrend)
            if rsi[i] < 30 and close[i] > ema_20_6h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought (>70) and price below weekly EMA20 (downtrend)
            elif rsi[i] > 70 and close[i] < ema_20_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: RSI overbought (>70) or price below weekly EMA20 (trend change)
            if rsi[i] > 70 or close[i] < ema_20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold (<30) or price above weekly EMA20 (trend change)
            if rsi[i] < 30 or close[i] > ema_20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals