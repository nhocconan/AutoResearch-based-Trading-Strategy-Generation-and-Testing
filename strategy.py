#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Trend_Following_with_Adaptive_Exit_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Weekly EMA34 for trend filter ===
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === Daily ATR(14) for volatility and exit ===
    tr = np.maximum(high - low, 
                    np.maximum(np.abs(high - np.roll(close, 1)), 
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === Daily EMA50 for exit filter ===
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(atr14[i]) or 
            np.isnan(ema50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price above weekly EMA34 and rising
            if close[i] > ema34_1w_aligned[i] and close[i] > close[i-1]:
                signals[i] = 0.25
                position = 1
            # Short entry: price below weekly EMA34 and falling
            elif close[i] < ema34_1w_aligned[i] and close[i] < close[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below EMA50 or trailing stop
            if close[i] < ema50[i] or close[i] < (high[i] - 1.5 * atr14[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above EMA50 or trailing stop
            if close[i] > ema50[i] or close[i] > (low[i] + 1.5 * atr14[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly EMA34 trend filter with daily EMA50/ATR-based exits.
# In bull markets: weekly EMA34 catches major uptrends, EMA50 exits on pullbacks.
# In bear markets: weekly EMA34 identifies downtrends, EMA50 exits on bounces.
# ATR-based trailing stops protect against reversals. Designed for 1d timeframe
# to target 20-60 trades over 4 years (5-15/year) minimizing fee drag.
# Uses discrete sizing (0.25) to reduce churn. Works on BTC/ETH via institutional
# weekly trend alignment. Simple 2-condition entry reduces overtrading risk.