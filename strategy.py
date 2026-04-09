#!/usr/bin/env python3
# 1h_mtf_cci_trend_reversal_v1
# Hypothesis: 1h strategy using Commodity Channel Index (CCI) for mean reversion entries with 4h/1d trend alignment. Enters long when CCI < -100 (oversold) and price > 4h EMA20 and 1d EMA50; short when CCI > 100 (overbought) and price < 4h EMA20 and 1d EMA50. Uses discrete position sizing (0.20) to limit fee drag and session filter (08-20 UTC) to reduce noise. Designed for low turnover (target: 15-37 trades/year) to work in both bull and bear markets by fading extremes only when aligned with higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_cci(high, low, close, period=20):
    """Calculate Commodity Channel Index"""
    if len(close) < period:
        return np.full_like(close, np.nan, dtype=float)
    tp = (high + low + close) / 3.0
    ma = pd.Series(tp).rolling(window=period, min_periods=period).mean()
    mad = pd.Series(tp).rolling(window=period, min_periods=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - ma) / (0.015 * mad)
    return cci.values

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

name = "1h_mtf_cci_trend_reversal_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate CCI on primary timeframe
    cci = calculate_cci(high, low, close, 20)
    
    # 4h HTF trend filter: EMA20
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    ema_20_4h = calculate_ema(df_4h['close'], 20)
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d HTF trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = calculate_ema(df_1d['close'], 50)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN or outside session
        if (np.isnan(cci[i]) or np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI crosses above -50 (mean reversion complete) or trend breaks
            if cci[i] > -50 or close[i] < ema_20_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: CCI crosses below 50 (mean reversion complete) or trend breaks
            if cci[i] < 50 or close[i] > ema_20_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter only with CCI extreme and both HTF trends aligned
            if cci[i] < -100 and close[i] > ema_20_4h_aligned[i] and close[i] > ema_50_1d_aligned[i]:
                # Oversold + bullish 4h + bullish 1d -> long
                position = 1
                signals[i] = 0.20
            elif cci[i] > 100 and close[i] < ema_20_4h_aligned[i] and close[i] < ema_50_1d_aligned[i]:
                # Overbought + bearish 4h + bearish 1d -> short
                position = -1
                signals[i] = -0.20
    
    return signals