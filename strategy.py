#!/usr/bin/env python3
# 4H_KAMA_14_RSI21_BullBear
# Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) to capture trend direction with RSI for momentum confirmation.
# Long when: KAMA slope > 0, RSI > 50, and price above KAMA.
# Short when: KAMA slope < 0, RSI < 50, and price below KAMA.
# Uses 1d trend filter: only trade in direction of daily EMA50 trend.
# Works in bull/bear by following trend and using momentum to confirm strength.
# Target: 20-40 trades/year per symbol.

name = "4H_KAMA_14_RSI21_BullBear"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (14-period)
    close_s = pd.Series(close)
    direction = np.abs(close_s.diff(14))
    volatility = close_s.diff().abs().rolling(window=14, min_periods=14).sum()
    er = np.where(volatility > 0, direction / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA slope (3-period change)
    kama_slope = np.diff(kama, prepend=kama[0])
    
    # RSI (21-period)
    delta = close_s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/21, adjust=False, min_periods=21).mean()
    ma_down = down.ewm(alpha=1/21, adjust=False, min_periods=21).mean()
    rsi = 100 - (100 / (1 + ma_up / (ma_down + 1e-10)))
    rsi = rsi.values
    
    # Daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align daily trend to 4h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(kama_slope[i]) or np.isnan(rsi[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_up = kama_slope[i] > 0
        kama_down = kama_slope[i] < 0
        rsi_above = rsi[i] > 50
        rsi_below = rsi[i] < 50
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: daily uptrend + KAMA up + RSI > 50 + price above KAMA
            if daily_up and kama_up and rsi_above and price_above_kama:
                signals[i] = 0.25
                position = 1
            # Enter short: daily downtrend + KAMA down + RSI < 50 + price below KAMA
            elif daily_down and kama_down and rsi_below and price_below_kama:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions: trend changes or momentum fades
            if not daily_up or not kama_up or not rsi_above or close[i] <= kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: trend changes or momentum fades
            if not daily_down or not kama_down or not rsi_below or close[i] >= kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals