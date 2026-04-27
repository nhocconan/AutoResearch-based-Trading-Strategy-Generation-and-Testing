#!/usr/bin/env python3
"""
Hypothesis: 6-hour Elder Ray Index with 12-hour trend filter and volume confirmation.
Elder Ray uses Bull Power (High - EMA) and Bear Power (Low - EMA) to measure bull/bear strength.
Long when Bull Power > 0 and rising + Bear Power < 0 + price above 12h EMA + volume > 1.5x average.
Short when Bear Power < 0 and falling + Bull Power < 0 + price below 12h EMA + volume > 1.5x average.
Uses 12h EMA for trend to avoid counter-trend trades. Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(data, period):
    """Exponential Moving Average"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    return pd.Series(data).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 6h EMA(13) for Elder Ray
    ema_13 = ema(close, 13)
    
    # Bull Power = High - EMA
    bull_power = high - ema_13
    
    # Bear Power = Low - EMA
    bear_power = low - ema_13
    
    # Align Elder Ray components to 6h timeframe
    ema_13_aligned = align_htf_to_ltf(prices, df_12h, ema_13)
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    
    # Calculate 12h EMA(21) for trend filter
    close_12h = df_12h['close'].values
    ema_21_12h = ema(close_12h, 21)
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Calculate 6h volume MA(20)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA(13), Bull/Bear Power, EMA(21)
    start_idx = max(13, 21, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(ema_21_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20[i]
        ema_13_now = ema_13_aligned[i]
        bull_power_now = bull_power_aligned[i]
        bear_power_now = bear_power_aligned[i]
        trend_12h = ema_21_12h_aligned[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Elder Ray signals
        bull_rising = bull_power_now > bull_power_aligned[i-1] if i > 0 else False
        bear_falling = bear_power_now < bear_power_aligned[i-1] if i > 0 else False
        
        # Entry conditions
        if position == 0:
            # Long: Bull Power > 0 and rising + Bear Power < 0 + above trend + volume
            if bull_power_now > 0 and bull_rising and bear_power_now < 0 and price_now > trend_12h and vol_filter:
                signals[i] = size
                position = 1
            # Short: Bear Power < 0 and falling + Bull Power < 0 + below trend + volume
            elif bear_power_now < 0 and bear_falling and bull_power_now < 0 and price_now < trend_12h and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power <= 0 or trend turns down
            if bull_power_now <= 0 or price_now < trend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Bear Power >= 0 or trend turns up
            if bear_power_now >= 0 or price_now > trend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0