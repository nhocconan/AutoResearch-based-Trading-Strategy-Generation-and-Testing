#/usr/bin/env python3
# 1d_1w_RSI_Extreme_Pullback_WeeklyTrend
# Hypothesis: On daily timeframe, buy RSI oversold pullbacks (RSI<30) in weekly uptrend,
# sell RSI overbought bounces (RSI>70) in weekly downtrend. Uses weekly trend filter to
# avoid counter-trend trades, reducing whipsaw in sideways markets. Designed for low
# trade frequency (<25/year) with high win rate via trend alignment and mean reversion
# within trend. Works in bull (buy dips) and bear (sell rallies) by following weekly trend.

name = "1d_1w_RSI_Extreme_Pullback_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly EMA(34) for trend
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough history for RSI and weekly EMA
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if weekly EMA is not ready
        if np.isnan(ema_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend: price above EMA = uptrend, below = downtrend
        # Need weekly close aligned to daily
        close_1w = df_1w['close'].values
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
        uptrend = close_1w_aligned[i] > ema_1w_aligned[i]
        downtrend = close_1w_aligned[i] < ema_1w_aligned[i]
        
        rsi_val = rsi_values[i]
        
        if position == 0:
            # Long: RSI oversold (<30) in weekly uptrend
            if rsi_val < 30 and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) in weekly downtrend
            elif rsi_val > 70 and downtrend:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: RSI returns to neutral (50) or trend fails
                if rsi_val >= 50 or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: RSI returns to neutral (50) or trend fails
                if rsi_val <= 50 or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals