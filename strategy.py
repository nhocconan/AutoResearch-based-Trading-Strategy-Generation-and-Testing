# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI2_Confirm_v3
Hypothesis: Use 1-day KAMA to determine trend direction, with RSI(2) for mean-reversion entries in the direction of the trend, and volume confirmation. This combines trend-following with short-term mean reversion to capture pullbacks in strong trends on the daily timeframe, reducing whipsaw and improving win rate. The strategy aims for low trade frequency (10-30 trades/year) by requiring alignment of trend, momentum, and volume, making it suitable for both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w HTF data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w trend filter: 50-period EMA ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === KAMA (1-day close) ===
    close = prices['close'].values
    direction = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(close) > 1 else 0
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (0.66 - 0.06) + 0.06) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(2) on 1-day close ===
    delta = np.diff(close, n=1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=2, min_periods=2).mean().values
    avg_loss = pd.Series(loss).rolling(window=2, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to match length (first value is NaN)
    rsi = np.concatenate([np.array([np.nan]), rsi])
    
    # === Volume confirmation: 20-period volume average ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(kama[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        vol_spike = vol_ratio[i]
        trend_1w = ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: Price > KAMA (uptrend) + RSI(2) < 15 (oversold) + volume spike > 1.5
            if (price_close > kama_val and 
                rsi_val < 15 and 
                vol_spike > 1.5 and
                price_close > trend_1w):
                signals[i] = 0.25
                position = 1
            # Short: Price < KAMA (downtrend) + RSI(2) > 85 (overbought) + volume spike > 1.5
            elif (price_close < kama_val and 
                  rsi_val > 85 and 
                  vol_spike > 1.5 and
                  price_close < trend_1w):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price crosses KAMA (trend reversal signal)
            if position == 1 and price_close < kama_val:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_Trend_RSI2_Confirm_v3"
timeframe = "1d"
leverage = 1.0