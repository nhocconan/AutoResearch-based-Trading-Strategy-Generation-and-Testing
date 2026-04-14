# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: On daily timeframe, combine 7-day Wilder's RSI with 20-day Bollinger Bands
to identify mean-reversion opportunities during low volatility squeezes, filtered by
weekly trend via 13-week EMA. Enter long when RSI < 30 and price touches lower BB
in an uptrend (weekly EMA slope up), short when RSI > 70 and price touches upper BB
in a downtrend. Exit on RSI crossing 50 or BB median touch. Uses discrete position
sizing (0.25) to limit risk and trades to ~10-20/year.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load weekly data ONCE before loop
    df_w = get_htf_data(prices, '1w')
    
    # Weekly EMA(13) for trend filter
    ema13_w = pd.Series(df_w['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_w_aligned = align_htf_to_ltf(prices, df_w, ema13_w)
    
    # Daily RSI(14) using Wilder's smoothing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily Bollinger Bands (20, 2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    mid = sma20
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position
    
    # Start after sufficient data
    start = 40
    
    for i in range(start, n):
        if (np.isnan(ema13_w_aligned[i]) or np.isnan(rsi[i]) or
            np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(mid[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        weekly_trend_up = ema13_w_aligned[i] > ema13_w_aligned[i-1]  # rising weekly EMA
        weekly_trend_down = ema13_w_aligned[i] < ema13_w_aligned[i-1]  # falling weekly EMA
        
        if position == 0:
            # Long: oversold + touches lower BB + weekly uptrend
            if rsi_val < 30 and price <= lower[i] and weekly_trend_up:
                position = 1
                signals[i] = position_size
            # Short: overbought + touches upper BB + weekly downtrend
            elif rsi_val > 70 and price >= upper[i] and weekly_trend_down:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses above 50 or price touches mid BB
            if rsi_val > 50 or price >= mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI crosses below 50 or price touches mid BB
            if rsi_val < 50 or price <= mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "daily_rsi_bb_weekly_trend_v1"
timeframe = "1d"
leverage = 1.0