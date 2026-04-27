#!/usr/bin/env python3
"""
Hypothesis: 4-hour price action above/below 1-day VWAP with volume confirmation and weekly trend filter.
Enters long when price crosses above VWAP with above-average volume and weekly uptrend.
Enters short when price crosses below VWAP with above-average volume and weekly downtrend.
Uses weekly timeframe for trend structure to reduce noise and avoid false signals.
Designed to work in both bull and bear markets by following the weekly trend while using
VWAP for mean reversion and volume for conviction. Target: 25-35 trades/year per
symbol to minimize fee drift and avoid overtrading.
"""

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
    
    # Get daily data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily VWAP: cumulative (price * volume) / cumulative volume
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    pv_1d = typical_price_1d * df_1d['volume'].values
    cum_pv_1d = np.cumsum(pv_1d)
    cum_vol_1d = np.cumsum(df_1d['volume'].values)
    vwap_1d = np.divide(cum_pv_1d, cum_vol_1d, out=np.zeros_like(cum_pv_1d), where=cum_vol_1d!=0)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate daily volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need VWAP, volume MA, and weekly EMA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(vwap_1d_aligned[i-1]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 4h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vwap_now = vwap_1d_aligned[i]
        vwap_prev = vwap_1d_aligned[i-1]
        vol_ma = vol_ma_20_1d_aligned[i]
        trend_1w = ema_20_1w_aligned[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Price/VWAP cross signals
        price_cross_above_vwap = price_now > vwap_now and close[i-1] <= vwap_prev
        price_cross_below_vwap = price_now < vwap_now and close[i-1] >= vwap_prev
        
        # Entry conditions
        if position == 0:
            # Long: price crosses above VWAP with volume + weekly uptrend
            if price_cross_above_vwap and vol_filter and price_now > trend_1w:
                signals[i] = size
                position = 1
            # Short: price crosses below VWAP with volume + weekly downtrend
            elif price_cross_below_vwap and vol_filter and price_now < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below VWAP or weekly trend turns down
            if price_cross_below_vwap or price_now < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above VWAP or weekly trend turns up
            if price_cross_above_vwap or price_now > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_VWAP_Cross_1dVolume_1wTrend"
timeframe = "4h"
leverage = 1.0