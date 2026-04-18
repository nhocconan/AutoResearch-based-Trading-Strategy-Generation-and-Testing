#!/usr/bin/env python3
"""
1h_1hr_4hTrend_1dVolatilityBreakout
Hypothesis: Use 4h EMA for trend direction and 1d ATR volatility to trigger breakout entries on 1h.
Long when price > 4h EMA(50) and breaks above 1h high + 0.5*1d ATR.
Short when price < 4h EMA(50) and breaks below 1h low - 0.5*1d ATR.
ATR filter ensures entries only during volatile regimes, reducing whipsaws.
Target: 15-30 trades/year (60-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 4h EMA(50) for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    tr1 = np.maximum(df_1d['high'], df_1d['close'].shift(1)) - np.minimum(df_1d['low'], df_1d['close'].shift(1))
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1h rolling high/low for breakout levels
    high_roll = pd.Series(high).rolling(window=2, min_periods=2).max().values  # previous bar high
    low_roll = pd.Series(low).rolling(window=2, min_periods=2).min().values   # previous bar low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(high_roll[i]) or np.isnan(low_roll[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_4h_aligned[i]
        atr_val = atr_1d_aligned[i]
        prev_high = high_roll[i]
        prev_low = low_roll[i]
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > 4h EMA and breaks above prior high + 0.5*ATR
            if price > ema_trend and price > prev_high + 0.5 * atr_val:
                signals[i] = 0.20
                position = 1
            # Short: price < 4h EMA and breaks below prior low - 0.5*ATR
            elif price < ema_trend and price < prev_low - 0.5 * atr_val:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long if price closes below 4h EMA or below prior low
            if price < ema_trend or price < prev_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short if price closes above 4h EMA or above prior high
            if price > ema_trend or price > prev_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_1hr_4hTrend_1dVolatilityBreakout"
timeframe = "1h"
leverage = 1.0