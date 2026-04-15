#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and session filter
# Long when RSI < 30 + price > 4h EMA50 + session (08-20 UTC)
# Short when RSI > 70 + price < 4h EMA50 + session (08-20 UTC)
# Uses discrete position sizing (0.20) to minimize fee churn.
# RSI mean reversion works in ranging markets; 4h EMA50 filter ensures we trade with higher timeframe trend.
# Session filter reduces noise during low-volume hours. Target: 15-30 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h Indicator: EMA(50) for trend filter ===
    close_4h = df_4h['close'].values
    ema_span = 50
    ema_4h = pd.Series(close_4h).ewm(span=ema_span, min_periods=ema_span, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === 1h Indicator: RSI(14) ===
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[rsi_period-1] = np.mean(gain[:rsi_period])
    avg_loss[rsi_period-1] = np.mean(loss[:rsi_period])
    
    for i in range(rsi_period, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(rsi_period, ema_span) + 5
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. RSI < 30 (oversold)
        # 2. Price > 4h EMA50 (uptrend filter)
        if (rsi[i] < 30) and (close[i] > ema_4h_aligned[i]):
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. RSI > 70 (overbought)
        # 2. Price < 4h EMA50 (downtrend filter)
        elif (rsi[i] > 70) and (close[i] < ema_4h_aligned[i]):
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_RSI14_4hEMA50_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0