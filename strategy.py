#!/usr/bin/env python3

"""
Hypothesis: 4-hour Volume Weighted Average Price (VWAP) with 1-day Exponential Moving Average (EMA) trend filter.
Trades long when price crosses above VWAP in an uptrend (price > daily EMA), short when price crosses below VWAP in a downtrend (price < daily EMA).
Uses VWAP as a dynamic intraday support/resistance level and daily EMA for higher timeframe trend bias.
Designed for low trade frequency (20-50 trades/year) by requiring both VWAP cross and trend alignment.
Works in both bull and bear markets by following the daily trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate VWAP typical price and cumulative values
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    
    # Cumulative sums for VWAP calculation
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(volume)
    
    # VWAP calculation (avoid division by zero)
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # Load daily data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA for trend filter (34-period)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if VWAP or EMA not ready
        if np.isnan(vwap[i]) or np.isnan(ema_34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Entry conditions: VWAP cross with trend alignment
        if position == 0:
            # Long: price crosses above VWAP with uptrend bias (price > daily EMA)
            if close[i] > vwap[i] and close[i-1] <= vwap[i-1] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below VWAP with downtrend bias (price < daily EMA)
            elif close[i] < vwap[i] and close[i-1] >= vwap[i-1] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price crosses back through VWAP
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below VWAP
                if close[i] < vwap[i] and close[i-1] >= vwap[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above VWAP
                if close[i] > vwap[i] and close[i-1] <= vwap[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_VWAP_EMA34_Trend_Follow"
timeframe = "4h"
leverage = 1.0