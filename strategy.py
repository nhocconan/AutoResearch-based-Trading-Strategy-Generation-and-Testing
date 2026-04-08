#!/usr/bin/env python3
"""
4h_mtf_cci_trend_reversal_v1
Hypothesis: Use Commodity Channel Index (CCI) on daily chart with trend filter from weekly chart.
- Long when CCI crosses above -100 from below with price above weekly EMA20 (uptrend)
- Short when CCI crosses below +100 from above with price below weekly EMA20 (downtrend)
- Uses mean-reversion within trend context to avoid counter-trend trades
- Designed for low trade frequency (20-50/year) with high win rate
- Works in bull/bear via trend filter and mean-reversion logic
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_mtf_cci_trend_reversal_v1"
timeframe = "4h"
leverage = 1.0

def calculate_cci(high, low, close, period=20):
    """Calculate Commodity Channel Index"""
    if len(high) < period:
        return np.full(len(high), np.nan)
    
    tp = (high + low + close) / 3.0
    ma = np.full_like(tp, np.nan)
    
    for i in range(period-1, len(tp)):
        ma[i] = np.mean(tp[i-period+1:i+1])
    
    mad = np.full_like(tp, np.nan)
    for i in range(period-1, len(tp)):
        mad[i] = np.mean(np.abs(tp[i-period+1:i+1] - ma[i]))
    
    cci = np.full_like(tp, np.nan)
    for i in range(period-1, len(tp)):
        if mad[i] != 0:
            cci[i] = (tp[i] - ma[i]) / (0.015 * mad[i])
    
    return cci

def calculate_ema(close, period):
    """Calculate EMA with proper handling"""
    if len(close) < period:
        return np.full_like(close, np.nan, dtype=float)
    
    ema = np.full_like(close, np.nan, dtype=float)
    alpha = 2.0 / (period + 1)
    ema[period-1] = np.mean(close[:period])
    for i in range(period, len(close)):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i-1]
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for CCI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily CCI
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    cci = calculate_cci(high_1d, low_1d, close_1d, 20)
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = calculate_ema(close_1w, 20)
    
    # Align indicators to 4h timeframe
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if np.isnan(cci_aligned[i]) or np.isnan(ema_20_1w_aligned[i]):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        cci_val = cci_aligned[i]
        ema_trend = ema_20_1w_aligned[i]
        price = close[i]
        
        if position == 1:  # Long
            # Exit: CCI crosses below +100 (overbought) or trend changes
            if cci_val < 100 or price < ema_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: CCI crosses above -100 (oversold) or trend changes
            if cci_val > -100 or price > ema_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: CCI crosses above -100 from below with uptrend
            if i > 50 and not np.isnan(cci_aligned[i-1]):
                if cci_aligned[i-1] <= -100 and cci_val > -100 and price > ema_trend:
                    position = 1
                    signals[i] = 0.25
            # Enter short: CCI crosses below +100 from above with downtrend
            elif i > 50 and not np.isnan(cci_aligned[i-1]):
                if cci_aligned[i-1] >= 100 and cci_val < 100 and price < ema_trend:
                    position = -1
                    signals[i] = -0.25
    
    return signals