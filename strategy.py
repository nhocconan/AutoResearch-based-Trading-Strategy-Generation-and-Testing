#!/usr/bin/env python3
"""
6h_LongOnly_GoldenCross_Strategy
Hypothesis: Captures long-term uptrends using a 21/55 EMA golden cross on the 6h chart,
filtered by 12h ADX > 25 to ensure trending conditions. Exits on death cross or ADX drop.
Designed for low trade frequency (~10-25/year) to minimize fee impact, works in bull markets
by capturing trends and avoids losses in bear markets by staying flat when ADX < 25.
"""

name = "6h_LongOnly_GoldenCross_Strategy"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 6h price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- 12h ADX for trend filter ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate ADX components on 12h data
    plus_dm = np.zeros(len(df_12h))
    minus_dm = np.zeros(len(df_12h))
    tr = np.zeros(len(df_12h))
    
    for i in range(1, len(df_12h)):
        high_diff = df_12h['high'].iloc[i] - df_12h['high'].iloc[i-1]
        low_diff = df_12h['low'].iloc[i-1] - df_12h['low'].iloc[i]
        plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
        minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
        tr[i] = max(
            df_12h['high'].iloc[i] - df_12h['low'].iloc[i],
            abs(df_12h['high'].iloc[i] - df_12h['close'].iloc[i-1]),
            abs(df_12h['low'].iloc[i] - df_12h['close'].iloc[i-1])
        )
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    period = 14
    atr_12h = wilder_smooth(tr, period)
    plus_di_12h = 100 * wilder_smooth(plus_dm, period) / atr_12h
    minus_di_12h = 100 * wilder_smooth(minus_dm, period) / atr_12h
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h + 1e-10)
    adx_12h = wilder_smooth(dx_12h, period)
    
    # Align 12h ADX to 6h
    adx_12h_6h = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # --- 6h EMA 21 and 55 for golden cross ---
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_55 = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Golden cross: EMA21 crosses above EMA55
    golden_cross = (ema_21 > ema_55) & (ema_21 <= ema_55)
    # Death cross: EMA21 crosses below EMA55
    death_cross = (ema_21 < ema_55) & (ema_21 >= ema_55)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(adx_12h_6h[i]) or np.isnan(ema_21[i]) or np.isnan(ema_55[i]):
            if position == 1:
                signals[i] = 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Only go long when ADX indicates strong trend (>25)
        strong_trend = adx_12h_6h[i] > 25
        
        if position == 0:
            # Enter long on golden cross during strong trend
            if golden_cross[i] and strong_trend:
                signals[i] = 0.25
                position = 1
        else:
            # Exit on death cross or when trend weakens (ADX < 20)
            if death_cross[i] or adx_12h_6h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals