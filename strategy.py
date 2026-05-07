#!/usr/bin/env python3
name = "1d_KAMA_Trend_With_RSI_Momentum"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly close for trend filter
    weekly_close = df_1w['close'].values
    
    # Daily KAMA (Kaufman Adaptive Moving Average) for trend direction
    # ER = Efficiency Ratio = abs(close - close[period]) / sum(abs(close - close[-1]))
    # SC = [ER * (fastest - slowest) + slowest]^2
    # KAMA = prevKAMA + SC * (price - prevKAMA)
    close_series = pd.Series(close)
    change = abs(close_series - close_series.shift(10))
    volatility = abs(close_series.diff()).rolling(window=10, min_periods=10).sum()
    er = change / volatility
    er = er.fillna(0)
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Daily RSI for momentum filter
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Neutral when undefined
    
    # Daily volume spike: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for KAMA and RSI warmup
    
    for i in range(start_idx, n):
        # Skip if weekly data not aligned yet
        weekly_idx = i // 7  # Approximate weekly index (7 days per week)
        if weekly_idx >= len(weekly_close):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        weekly_close_val = weekly_close[weekly_idx]
        
        if np.isnan(kama[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA, RSI > 55, weekly close above weekly KAMA, volume spike
            weekly_kama = 0
            if weekly_idx >= 10:  # Need enough data for weekly KAMA
                weekly_close_series = pd.Series(weekly_close[:weekly_idx+1])
                weekly_change = abs(weekly_close_series - weekly_close_series.shift(10))
                weekly_volatility = abs(weekly_close_series.diff()).rolling(window=10, min_periods=10).sum()
                weekly_er = weekly_change / weekly_volatility
                weekly_er = weekly_er.fillna(0)
                weekly_fast_sc = 2 / (2 + 1)
                weekly_slow_sc = 2 / (30 + 1)
                weekly_sc = (weekly_er * (weekly_fast_sc - weekly_slow_sc) + weekly_slow_sc) ** 2
                weekly_kama_vals = np.zeros(len(weekly_close))
                weekly_kama_vals[0] = weekly_close[0]
                for j in range(1, len(weekly_close)):
                    weekly_kama_vals[j] = weekly_kama_vals[j-1] + weekly_sc[j] * (weekly_close[j] - weekly_kama_vals[j-1])
                weekly_kama = weekly_kama_vals[weekly_idx]
            
            if close[i] > kama[i] and rsi[i] > 55 and weekly_close_val > weekly_kama and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA, RSI < 45, weekly close below weekly KAMA, volume spike
            elif close[i] < kama[i] and rsi[i] < 45 and weekly_close_val < weekly_kama and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below KAMA or RSI < 40
            if close[i] < kama[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above KAMA or RSI > 60
            if close[i] > kama[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily trend following with KAMA adaptive trend filter and RSI momentum filter.
# Long when price > daily KAMA, RSI > 55 (bullish momentum), weekly close > weekly KAMA (bullish weekly trend), and volume spike confirms.
# Short when price < daily KAMA, RSI < 45 (bearish momentum), weekly close < weekly KAMA (bearish weekly trend), and volume spike confirms.
# Uses weekly timeframe for trend alignment to avoid whipsaws, daily for entry timing.
# KAMA adapts to market noise - fast in trends, slow in ranging markets.
# Volume spike (>1.5x average) ensures conviction. Discrete 0.25 position size limits risk.
# Works in bull markets (trend + momentum) and bear markets (reverse criteria).
# Target: 15-25 trades/year to minimize fee drag while capturing sustained moves.