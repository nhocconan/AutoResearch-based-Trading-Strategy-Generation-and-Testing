#!/usr/bin/env python3
# 4H_RSI_Extremes_Bollinger_Trend_Filter
# Hypothesis: RSI extremes (overbought/oversold) with Bollinger Band mean reversion,
# filtered by 1-day EMA trend for direction. Works in bull (buy dips in uptrend) and
# bear (sell rallies in downtrend) by using trend filter. Target: 20-40 trades/year.

name = "4H_RSI_Extremes_Bollinger_Trend_Filter"
timeframe = "4h"
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
    
    # === 1d Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 34-period EMA on 1d for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Bollinger Bands (20, 2.0) ===
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20 + 2.0 * std_20
    lower_bb = ma_20 - 2.0 * std_20
    
    # === RSI (14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # LONG: RSI oversold (<30) and price near lower BB in uptrend
            if (rsi[i] < 30 and close[i] <= lower_bb[i] and trend_up):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought (>70) and price near upper BB in downtrend
            elif (rsi[i] > 70 and close[i] >= upper_bb[i] and trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: RSI returns to neutral (50) or trend changes
            if (rsi[i] >= 50 or not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral (50) or trend changes
            if (rsi[i] <= 50 or not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals