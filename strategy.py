#!/usr/bin/env python3
# 1d_KAMA_Trend_RSI_Filter
# Hypothesis: On 1d timeframe, use KAMA (Kaufman Adaptive Moving Average) to capture medium-term trend.
# Enter long when price > KAMA and RSI > 50 (bullish momentum).
# Enter short when price < KAMA and RSI < 50 (bearish momentum).
# Exit when price crosses KAMA in opposite direction.
# Uses weekly trend filter to avoid counter-trend trades, targeting 10-20 trades/year for low friction.
# Works in bull via KAMA uptrend and in bear via KAMA downtrend with RSI filter.

name = "1d_KAMA_Trend_RSI_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA calculation (10-day ER, 2/30 smoothing)
    price_diff = np.abs(np.diff(close, prepend=close[0]))
    direction = np.abs(close - np.roll(close, 10))
    volatility = np.sum(price_diff.reshape(-1, 10), axis=1)
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(weekly_ema50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        kama_val = kama[i]
        rsi_val = rsi[i]
        weekly_trend = weekly_ema50_aligned[i]
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # LONG: Price above KAMA, RSI > 50, and weekly uptrend
            if close[i] > kama_val and rsi_val > 50 and close[i] > weekly_trend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA, RSI < 50, and weekly downtrend
            elif close[i] < kama_val and rsi_val < 50 and close[i] < weekly_trend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA
            if close[i] < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA
            if close[i] > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals