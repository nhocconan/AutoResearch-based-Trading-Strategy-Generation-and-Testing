#!/usr/bin/env python3
# 1h_4h_1d_rsi_ema_trend_v1
# Strategy: 1h RSI + EMA trend following with 4h/1d trend filter
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: In trending markets (identified by 4h EMA20>EMA50 and 1d close>SMA50),
# use 1h EMA20/50 crossovers with RSI(14)>50 for longs and <50 for shorts.
# In ranging markets (4h EMA20<EMA50 or 1d close<SMA50), use 1h RSI reversals at
# 30/70 levels. Position size fixed at 0.20 to manage drawdown. Session filter
# (08-20 UTC) reduces noise. Designed for 15-30 trades/year to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_rsi_ema_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA20 and EMA50 for trend filter
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d close and SMA50 for trend filter
    sma_50_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # 1h EMA20 and EMA50 for entry signals
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(ema_20[i]) or np.isnan(ema_50[i]) or np.isnan(rsi[i]) or \
           np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or \
           np.isnan(sma_50_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Trend determination: 4h EMA20>EMA50 AND 1d close>SMA50 = uptrend
        is_uptrend = ema_20_4h_aligned[i] > ema_50_4h_aligned[i] and close[i] > sma_50_1d_aligned[i]
        # Downtrend: 4h EMA20<EMA50 AND 1d close<SMA50
        is_downtrend = ema_20_4h_aligned[i] < ema_50_4h_aligned[i] and close[i] < sma_50_1d_aligned[i]
        
        # 1h EMA crossover
        ema_cross_up = ema_20[i] > ema_50[i] and ema_20[i-1] <= ema_50[i-1]
        ema_cross_down = ema_20[i] < ema_50[i] and ema_20[i-1] >= ema_50[i-1]
        
        if in_session:
            if is_uptrend:
                # In uptrend: long on EMA bullish crossover with RSI>50
                if ema_cross_up and rsi[i] > 50 and position != 1:
                    position = 1
                    signals[i] = 0.20
                # Exit long on EMA bearish crossover
                elif position == 1 and ema_cross_down:
                    position = 0
                    signals[i] = 0.0
            elif is_downtrend:
                # In downtrend: short on EMA bearish crossover with RSI<50
                if ema_cross_down and rsi[i] < 50 and position != -1:
                    position = -1
                    signals[i] = -0.20
                # Exit short on EMA bullish crossover
                elif position == -1 and ema_cross_up:
                    position = 0
                    signals[i] = 0.0
            else:
                # Mixed or no clear trend: use RSI reversals at extremes
                # Long when RSI crosses above 30 from below
                if rsi[i] > 30 and rsi[i-1] <= 30 and position != 1:
                    position = 1
                    signals[i] = 0.20
                # Short when RSI crosses below 70 from above
                elif rsi[i] < 70 and rsi[i-1] >= 70 and position != -1:
                    position = -1
                    signals[i] = -0.20
                # Exit on opposite RSI cross
                elif position == 1 and rsi[i] < 70 and rsi[i-1] >= 70:
                    position = 0
                    signals[i] = 0.0
                elif position == -1 and rsi[i] > 30 and rsi[i-1] <= 30:
                    position = 0
                    signals[i] = 0.0
        else:
            # Outside session: flatten
            if position != 0:
                position = 0
                signals[i] = 0.0
    
    return signals