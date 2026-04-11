#!/usr/bin/env python3
# 1d_1w_keltner_trend_reversion_v2
# Strategy: 1-day Keltner Channel mean reversion with 1-week trend filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Price reversions from Keltner Channel extremes (2.0 ATR) combined with
# weekly EMA(21) trend filter capture mean-reversion opportunities in both bull and bear markets.
# Weekly trend ensures trades align with higher timeframe direction, reducing false signals.
# Uses discrete position sizing (0.25) to minimize fee churn.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_keltner_trend_reversion_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # 1-day ATR(10) for Keltner Channel
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # 1-day EMA(20) for Keltner Channel midline
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bands (2.0 ATR)
    keltner_upper = ema_20 + 2.0 * atr
    keltner_lower = ema_20 - 2.0 * atr
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after EMA/ATR warmup
        # Skip if any required data is invalid
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(ema_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Mean reversion signals: price at Keltner extremes
        touch_upper = close[i] >= keltner_upper[i]
        touch_lower = close[i] <= keltner_lower[i]
        
        # Trend filter: price above/below weekly EMA21
        uptrend = close[i] > ema_21_1w_aligned[i]
        downtrend = close[i] < ema_21_1w_aligned[i]
        
        # Entry logic: reversion from extreme + trend alignment
        if touch_lower and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif touch_upper and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price returns to midline (EMA20)
        elif position == 1 and close[i] <= ema_20[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= ema_20[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals