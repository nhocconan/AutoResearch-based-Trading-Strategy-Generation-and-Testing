#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and session filter.
# Long when price > 1h EMA50 AND 4h close > 4h EMA200 (bullish trend) AND hour between 08-20 UTC
# Short when price < 1h EMA50 AND 4h close < 4h EMA200 (bearish trend) AND hour between 08-20 UTC
# Exit when price crosses back through 1h EMA50
# Uses 4h EMA200 for trend direction (avoid counter-trend trades), 1h EMA50 for entry/exit timing.
# Session filter reduces noise during low-liquidity hours.
# Target: 15-35 trades/year per symbol.
name = "1h_EMA50_4hEMA200_Trend_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h EMA200 for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema200_4h = pd.Series(df_4h['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # Calculate 1h EMA50 for entry/exit
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # already datetime64[ms], .hour works
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200 to be ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema200_4h_aligned[i]) or np.isnan(ema50[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema200 = ema200_4h_aligned[i]
        ema50_val = ema50[i]
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long entry: price above EMA50 AND 4h trend bullish AND in session
            if price > ema50_val and close[i] > ema200 and in_session:
                signals[i] = 0.20
                position = 1
            # Short entry: price below EMA50 AND 4h trend bearish AND in session
            elif price < ema50_val and close[i] < ema200 and in_session:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA50
            if price < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses above EMA50
            if price > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals