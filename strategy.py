#!/usr/bin/env python3
"""
1h_4d_rsi_trend_v1
Strategy: 1h RSI mean reversion with 4h trend filter
Timeframe: 1h
Leverage: 1.0
Hypothesis: In 4h uptrend (price > EMA50), buy when 1h RSI < 30; in 4h downtrend (price < EMA50), sell when 1h RSI > 70. Uses 4h EMA50 for trend direction and 1h RSI for mean-reversion entries. Works in both bull and bear markets by following the higher timeframe trend while exploiting short-term overextensions on the 1h chart. Low-frequency design targets 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_rsi_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load higher timeframe data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08-20 UTC (major sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        price_close = close[i]
        ema_4h = ema_50_4h_aligned[i]
        rsi_val = rsi[i]
        
        # Trend filters
        uptrend = price_close > ema_4h
        downtrend = price_close < ema_4h
        
        # Entry conditions
        long_signal = uptrend and (rsi_val < 30)
        short_signal = downtrend and (rsi_val > 70)
        
        # Exit when RSI returns to neutral zone (40-60)
        exit_long = position == 1 and (rsi_val > 40)
        exit_short = position == -1 and (rsi_val < 60)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals

# Hypothesis: In 4h uptrend (price > EMA50), buy when 1h RSI < 30; in 4h downtrend (price < EMA50), sell when 1h RSI > 70. Uses 4h EMA50 for trend direction and 1h RSI for mean-reversion entries. Works in both bull and bear markets by following the higher timeframe trend while exploiting short-term overextensions on the 1h chart. Low-frequency design targets 15-30 trades/year to minimize fee drag.