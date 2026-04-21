#!/usr/bin/env python3
"""
1h_Volume_Weighted_RSI_With_4h_Trend_Filter
Hypothesis: Combines 1h volume-weighted RSI (VW-RSI) with 4h EMA50 trend filter to capture momentum in trending markets while avoiding sideways chop. Volume weighting gives more importance to price moves on high volume, making RSI more responsive to institutional activity. The 4h EMA50 ensures we only trade in the direction of the higher timeframe trend, reducing false signals during corrections. Discrete sizing (0.20) and session filter (08-20 UTC) minimize fee drag. Target: 80-120 total trades over 4 years (20-30/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # === Load HTF data ONCE before loop (4h for EMA trend) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # === 4h EMA50 for trend filter ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1h Volume-Weighted RSI (14-period) ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Typical price
    typical_price = (high + low + close) / 3.0
    
    # Volume-weighted typical price change
    vwtp = typical_price * volume
    
    # Calculate changes
    delta = np.diff(vwtp, prepend=vwtp[0])
    
    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    
    # Smoothed average gains/losses (Wilder's smoothing)
    avg_gain = pd.Series(gains).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(losses).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # RSI calculation
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # === Session filter: 08-20 UTC (precompute hours array) ===
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):  # Start after warmup for RSI
        # Skip if indicators not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema_trend = ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: RSI > 55 (bullish momentum) AND price above 4h EMA50 (uptrend)
            long_condition = (rsi_val > 55.0) and (price > ema_trend)
            # Short: RSI < 45 (bearish momentum) AND price below 4h EMA50 (downtrend)
            short_condition = (rsi_val < 45.0) and (price < ema_trend)
            
            if long_condition:
                signals[i] = 0.20
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit long: RSI < 40 (loss of momentum) OR price below 4h EMA50 (trend change)
            if (rsi_val < 40.0) or (price < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI > 60 (loss of bearish momentum) OR price above 4h EMA50 (trend change)
            if (rsi_val > 60.0) or (price > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Volume_Weighted_RSI_With_4h_Trend_Filter"
timeframe = "1h"
leverage = 1.0