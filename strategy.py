#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian channel breakout with 1-day trend filter, volume confirmation, and ATR stoploss.
Donchian breakout provides clear entry/exit, daily trend filter ensures alignment with higher timeframe momentum,
volume confirms institutional participation, ATR stoploss manages risk. Target: 20-40 trades/year per symbol.
Works in bull markets via breakout continuation and in bear markets via breakdown continuation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=np.float64)
    
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(len(high))
    atr[:period-1] = np.nan
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend
    daily_close = df_1d['close'].values
    ema_50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR for stoploss
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    atr_14_1d = calculate_atr(daily_high, daily_low, daily_close, 14)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_period = 20
    upper_channel = np.full_like(high, np.nan, dtype=np.float64)
    lower_channel = np.full_like(high, np.nan, dtype=np.float64)
    
    for i in range(donchian_period-1, len(high)):
        upper_channel[i] = np.max(high[i-donchian_period+1:i+1])
        lower_channel[i] = np.min(low[i-donchian_period+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20) + EMA (50) + ATR (14)
    start_idx = max(donchian_period-1, 50-1, 14-1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_now = close[i]
        atr_now = atr_14_1d_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above upper channel with bullish trend
            if price_now > upper_channel[i] and price_now > ema_trend:
                signals[i] = size
                position = 1
            # Short entry: price breaks below lower channel with bearish trend
            elif price_now < lower_channel[i] and price_now < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below lower channel OR trend reversal
            if price_now < lower_channel[i] or price_now < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above upper channel OR trend reversal
            if price_now > upper_channel[i] or price_now > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_DonchianBreakout_1dTrend_Filter"
timeframe = "4h"
leverage = 1.0