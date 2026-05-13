#!/usr/bin/env python3
"""
4h_MeanReversion_With_Trend_Protection
Hypothesis: Buy when price is significantly below VWAP in a downtrend with high volume (panic), sell when price returns to VWAP. Short when price is significantly above VWAP in an uptrend with high volume (euphoria), cover when price returns to VWAP. Uses 1d trend filter to align with higher timeframe bias and avoid counter-trend trades. VWAP calculated from session open (00:00 UTC) to current bar. Target: 20-40 trades/year per symbol.
"""

name = "4h_MeanReversion_With_Trend_Protection"
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
    
    # Calculate VWAP from session start (00:00 UTC) for each day
    # Since we don't have datetime in values, we'll use a rolling window approximation
    # VWAP approximation: cumulative (price * volume) / cumulative volume, reset when volume is low (new session)
    # Simpler: use typical price * volume for VWAP calculation
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    
    # Cumulative VWAP with reset condition (when volume is very low, indicating new session)
    # We'll use a 24-period lookback for VWAP (24 * 4h = 4 days, but we reset based on volume)
    # Instead, use a rolling VWAP with period=24 to approximate session VWAP
    vwap = np.zeros(n)
    vol_sum = np.zeros(n)
    pv_sum = np.zeros(n)
    
    # Use expanding window with reset when volume drops significantly (new session)
    # Simpler approach: use 24-period rolling VWAP
    for i in range(n):
        if i == 0:
            pv_sum[i] = pv[i]
            vol_sum[i] = volume[i]
        else:
            # Reset if current volume is very low compared to recent average (new session)
            if i >= 24 and volume[i] < 0.1 * np.mean(volume[max(0, i-24):i]):
                pv_sum[i] = pv[i]
                vol_sum[i] = volume[i]
            else:
                pv_sum[i] = pv_sum[i-1] + pv[i]
                vol_sum[i] = vol_sum[i-1] + volume[i]
        
        if vol_sum[i] > 0:
            vwap[i] = pv_sum[i] / vol_sum[i]
        else:
            vwap[i] = typical_price[i]
    
    # Deviation from VWAP as percentage
    dev_pct = (close - vwap) / vwap
    
    # 4h trend: EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_4h = close > ema_50
    downtrend_4h = close < ema_50
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average (to avoid noise)
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        dev = dev_pct[i]
        uptrend = uptrend_4h[i]
        downtrend = downtrend_4h[i]
        uptrend_htf = uptrend_1d_aligned[i]
        downtrend_htf = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: price significantly below VWAP (-2%), in 4h downtrend (panic sell), 1d uptrend (higher timeframe bullish), volume confirmation
            if dev < -0.02 and downtrend and uptrend_htf and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: price significantly above VWAP (+2%), in 4h uptrend (euphoria buy), 1d downtrend (higher timeframe bearish), volume confirmation
            elif dev > 0.02 and uptrend and downtrend_htf and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to VWAP (within 0.5%) or 4h trend turns up
            if abs(dev) < 0.005 or uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to VWAP (within 0.5%) or 4h trend turns down
            if abs(dev) < 0.005 or downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals