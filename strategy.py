#!/usr/bin/env python3
"""
Hypothesis: 6-hour Volume-Weighted Average Price (VWAP) deviation with 1-week trend filter and volume confirmation.
In bull market (price > 1-week EMA50): long when price deviates below VWAP by >1.5 standard deviations and volume > 1.3x average.
In bear market (price < 1-week EMA50): short when price deviates above VWAP by >1.5 standard deviations and volume > 1.3x average.
VWAP acts as dynamic support/resistance, weekly trend filters direction, volume confirms institutional participation.
Target: 12-35 trades/year per symbol.
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
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend
    weekly_close = df_1w['close'].values
    ema_50_1w = np.empty_like(weekly_close, dtype=np.float64)
    ema_50_1w.fill(np.nan)
    if len(weekly_close) >= 50:
        alpha = 2.0 / (50 + 1)
        ema_50_1w[49] = np.mean(weekly_close[:50])
        for i in range(50, len(weekly_close)):
            ema_50_1w[i] = alpha * weekly_close[i] + (1 - alpha) * ema_50_1w[i-1]
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get weekly data for volume confirmation
    vol_1w = df_1w['volume'].values
    vol_ma_20_1w = np.empty_like(vol_1w, dtype=np.float64)
    vol_ma_20_1w.fill(np.nan)
    for i in range(19, len(vol_1w)):
        vol_ma_20_1w[i] = np.mean(vol_1w[i-19:i+1])
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    # Calculate 6-hour VWAP and standard deviation
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(typical_price, np.nan), where=vwap_den!=0)
    
    # Calculate rolling standard deviation of price-VWAP deviation
    price_dev = typical_price - vwap
    vwap_std = np.full_like(price_dev, np.nan)
    for i in range(19, len(price_dev)):  # 20-period std
        if not np.isnan(price_dev[i-19:i+1]).any():
            vwap_std[i] = np.std(price_dev[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need VWAP std (20), weekly EMA50 (50), weekly volume MA20 (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vwap[i]) or np.isnan(vwap_std[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        tp_now = typical_price[i]
        vol_ma = vol_ma_20_1w_aligned[i]
        
        # Current indicators
        vwap_val = vwap[i]
        vwap_std_val = vwap_std[i]
        ema_trend = ema_50_1w_aligned[i]
        
        # Weekly close price for trend comparison
        weekly_close_price = df_1w['close'].values
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close_price)
        if np.isnan(weekly_close_aligned[i]):
            signals[i] = 0.0
            continue
        weekly_close_val = weekly_close_aligned[i]
        
        # Volume filter: volume > 1.3x weekly average
        vol_filter = vol_now > 1.3 * vol_ma
        
        # VWAP deviation signals
        if vwap_std_val > 0:
            dev_below = (vwap_val - tp_now) > (1.5 * vwap_std_val)  # Price below VWAP
            dev_above = (tp_now - vwap_val) > (1.5 * vwap_std_val)  # Price above VWAP
        else:
            dev_below = False
            dev_above = False
        
        if position == 0:
            # Bull market (price > weekly EMA50): look for long when price deviates below VWAP
            if weekly_close_val > ema_trend and dev_below and vol_filter:
                signals[i] = size
                position = 1
            # Bear market (price < weekly EMA50): look for short when price deviates above VWAP
            elif weekly_close_val < ema_trend and dev_above and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to VWAP or trend changes to bear
            if tp_now >= vwap_val or weekly_close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to VWAP or trend changes to bull
            if tp_now <= vwap_val or weekly_close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_VWAP_Deviation_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0