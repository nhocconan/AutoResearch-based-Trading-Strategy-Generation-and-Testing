#!/usr/bin/env python3
# 1d_weekly_pullback_v1
# Hypothesis: Weekly trend + daily mean reversion on weekly pullbacks. Long when weekly EMA50 up, price pulls back to weekly EMA20 on daily timeframe with RSI < 40. Short when weekly EMA50 down, price pulls back to weekly EMA20 with RSI > 60. Uses weekly trend filter to avoid counter-trend trades, targeting 15-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_pullback_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Weekly EMA50 for trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Daily RSI(14) for mean reversion
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Wait for weekly EMA50
    
    for i in range(start_idx, n):
        # Skip if weekly data not available
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(ema20_1w_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price crosses above weekly EMA20 or RSI > 60
            if close[i] > ema20_1w_aligned[i] or rsi[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses below weekly EMA20 or RSI < 40
            if close[i] < ema20_1w_aligned[i] or rsi[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Weekly EMA50 up, price below weekly EMA20, RSI < 40
            if (ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and  # Weekly EMA50 rising
                close[i] < ema20_1w_aligned[i] and 
                rsi[i] < 40):
                position = 1
                signals[i] = 0.25
            # Short entry: Weekly EMA50 down, price above weekly EMA20, RSI > 60
            elif (ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and  # Weekly EMA50 falling
                  close[i] > ema20_1w_aligned[i] and 
                  rsi[i] > 60):
                position = -1
                signals[i] = -0.25
    
    return signals