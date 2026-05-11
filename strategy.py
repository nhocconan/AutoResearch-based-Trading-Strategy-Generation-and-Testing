#!/usr/bin/env python3
name = "1d_WeeklyTrend_DailyVWAP_Reversion"
timeframe = "1d"
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
    
    # Weekly trend: 10-week EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Daily VWAP (typical price * volume) cumulative
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, close)
    
    # Daily momentum: 5-day ROC
    roc_5 = np.zeros_like(close)
    roc_5[5:] = (close[5:] - close[:-5]) / close[:-5] * 100.0
    
    # Volume filter: current volume > 1.2x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.2 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Need enough data for VWAP and ROC
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_10_1w_aligned[i]) or np.isnan(vwap[i]) or
            np.isnan(roc_5[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above weekly EMA (uptrend) + price below VWAP (mean reversion) +
            # positive ROC + volume confirmation
            if close[i] > ema_10_1w_aligned[i] and close[i] < vwap[i] and roc_5[i] > 0 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly EMA (downtrend) + price above VWAP (mean reversion) +
            # negative ROC + volume confirmation
            elif close[i] < ema_10_1w_aligned[i] and close[i] > vwap[i] and roc_5[i] < 0 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses above VWAP OR weekly trend turns down
            if close[i] >= vwap[i] or close[i] < ema_10_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses below VWAP OR weekly trend turns up
            if close[i] <= vwap[i] or close[i] > ema_10_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals