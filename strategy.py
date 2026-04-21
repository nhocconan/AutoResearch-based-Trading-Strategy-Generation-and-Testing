#!/usr/bin/env python3
"""
12h_WVAF_Momentum_Breakout_V1
Hypothesis: Combine Williams %R momentum with Volume-Weighted Average Price (VWAP) on 12h timeframe.
Enter long when price crosses above VWAP and Williams %R exits oversold (< -80), short when price crosses below VWAP and Williams %R exits overbought (> -20).
Use 1-week trend filter: only take trades aligned with weekly EMA20 direction.
Includes ATR-based stop loss via signal=0 when price moves against position by 2*ATR.
Designed for low trade frequency (target 15-30/year) to minimize fee drag while capturing momentum shifts in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_r(high, low, close, period=14):
    """Calculate Williams %R"""
    highest_high = np.maximum.accumulate(high)
    lowest_low = np.minimum.accumulate(low)
    wr = np.full_like(high, np.nan)
    for i in range(len(high)):
        start = max(0, i - period + 1)
        hh = np.max(high[start:i+1])
        ll = np.min(low[start:i+1])
        if hh != ll:
            wr[i] = (hh - close[i]) / (hh - ll) * -100
        else:
            wr[i] = -50
    return wr

def calculate_vwap(high, low, close, volume):
    """Calculate Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3
    vwap = np.full_like(close, np.nan)
    cum_tpv = np.zeros(len(close))
    cum_vol = np.zeros(len(close))
    for i in range(len(close)):
        cum_tpv[i] = cum_tpv[i-1] + typical_price[i] * volume[i] if i > 0 else typical_price[i] * volume[i]
        cum_vol[i] = cum_vol[i-1] + volume[i] if i > 0 else volume[i]
        if cum_vol[i] > 0:
            vwap[i] = cum_tpv[i] / cum_vol[i]
    return vwap

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr = np.full_like(high, np.nan)
    for i in range(len(high)):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full_like(high, np.nan)
    for i in range(len(high)):
        if i < period - 1:
            atr[i] = np.nan
        else:
            start = i - period + 1
            atr[i] = np.mean(tr[start:i+1])
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Load 1w data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = np.full_like(close_1w, np.nan)
    alpha = 2 / (20 + 1)
    for i in range(len(close_1w)):
        if i == 0:
            ema_20_1w[i] = close_1w[i]
        elif not np.isnan(close_1w[i]):
            ema_20_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_20_1w[i-1]
        else:
            ema_20_1w[i] = ema_20_1w[i-1]
    
    # Align weekly EMA to 12h
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 12h data for signals
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate indicators
    williams_r = calculate_williams_r(high, low, close, 14)
    vwap = calculate_vwap(high, low, close, volume)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if np.isnan(ema_20_1w_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vwap[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_20 = ema_20_1w_aligned[i]
        wr = williams_r[i]
        vwap_val = vwap[i]
        atr_val = atr[i]
        
        # Trend filter: only trade in direction of weekly EMA20
        trend_up = price > ema_20
        trend_down = price < ema_20
        
        if position == 0:
            # Long: price crosses above VWAP, WR exits oversold, uptrend
            if price > vwap_val and williams_r[i-1] <= -80 and wr > -80 and trend_up:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price crosses below VWAP, WR exits overbought, downtrend
            elif price < vwap_val and williams_r[i-1] >= -20 and wr < -20 and trend_down:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss or reversal signal
            if price < entry_price - 2.0 * atr_val:  # stop loss
                signals[i] = 0.0
                position = 0
            elif price < vwap_val and wr < -50:  # momentum loss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss or reversal signal
            if price > entry_price + 2.0 * atr_val:  # stop loss
                signals[i] = 0.0
                position = 0
            elif price > vwap_val and wr > -50:  # momentum loss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WVAF_Momentum_Breakout_V1"
timeframe = "12h"
leverage = 1.0