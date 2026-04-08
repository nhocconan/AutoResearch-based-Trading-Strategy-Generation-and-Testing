#!/usr/bin/env python3
"""
1d_1w_momentum_regime_v1
Hypothesis: On daily timeframe, combine 50-day EMA trend with 14-day RSI momentum and weekly trend filter.
Enter long when price > EMA50, RSI > 55, and weekly close > weekly EMA20 (bullish alignment).
Enter short when price < EMA50, RSI < 45, and weekly close < weekly EMA20 (bearish alignment).
Use volatility filter to avoid choppy markets. Target 10-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_momentum_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily indicators
    # EMA(50) for trend
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly trend: close > EMA(20)
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_bullish = close_1w > ema_20_1w
    weekly_bearish = close_1w < ema_20_1w
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Volatility filter: ATR(14) > 1% of price to avoid chop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    vol_filter = atr > (close * 0.01)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_50[i]) or np.isnan(rsi[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: trend breaks down or momentum fades
            if close[i] <= ema_50[i] or rsi[i] < 45 or weekly_bearish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: trend breaks up or momentum fades
            if close[i] >= ema_50[i] or rsi[i] > 55 or weekly_bullish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: bullish alignment
            if (close[i] > ema_50[i] and rsi[i] > 55 and 
                weekly_bullish_aligned[i] > 0.5 and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: bearish alignment
            elif (close[i] < ema_50[i] and rsi[i] < 45 and 
                  weekly_bearish_aligned[i] > 0.5 and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals