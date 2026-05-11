#!/usr/bin/env python3
"""
1d_WeeklyVWAP_Rebound_Trend
Hypothesis: On the daily timeframe, buy when price pulls back to weekly VWAP during uptrend (price > weekly EMA50) and sell when price rallies to weekly VWAP during downtrend (price < weekly EMA50). Uses weekly VWAP as dynamic support/resistance and weekly EMA50 for trend filter. Designed for low-frequency trading (target 10-30 trades/year) to minimize fee drag. Works in bull markets (buy VWAP bounces in uptrend) and bear markets (sell VWAP rejections in downtrend).
"""

name = "1d_WeeklyVWAP_Rebound_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for VWAP and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly typical price and VWAP
    tp = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    tp_values = tp.values
    vol = df_1w['volume'].values
    
    # Cumulative TP*Volume and Volume for VWAP
    cum_tpv = np.cumsum(tp_values * vol)
    cum_vol = np.cumsum(vol)
    # Avoid division by zero
    vwap = np.where(cum_vol != 0, cum_tpv / cum_vol, tp_values)
    
    # Weekly EMA 50 for trend
    ema_50 = pd.Series(df_1w['close']).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    
    # Align to daily timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(vwap_aligned[i]) or np.isnan(ema_50_aligned[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price at or above weekly VWAP AND above weekly EMA50 (uptrend)
            if close[i] >= vwap_aligned[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price at or below weekly VWAP AND below weekly EMA50 (downtrend)
            elif close[i] <= vwap_aligned[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price crosses weekly VWAP in opposite direction
            if position == 1:
                # Exit long: price crosses below weekly VWAP
                if close[i] < vwap_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above weekly VWAP
                if close[i] > vwap_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals