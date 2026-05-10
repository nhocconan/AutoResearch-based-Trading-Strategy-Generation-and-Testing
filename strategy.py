#!/usr/bin/env python3
# 1h_OrderFlow_Imbalance_4hTrend_1dVolFilter
# Hypothesis: Use 4h EMA50 for trend direction, 1h volume-weighted price deviation for entry timing, and 1d volume filter to avoid low-liquidity noise. Designed for 1h to achieve 15-37 trades/year in both bull and bear markets by combining trend alignment with mean-reversion entries during high-volume periods.

name = "1h_OrderFlow_Imbalance_4hTrend_1dVolFilter"
timeframe = "1h"
leverage = 1.0

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
    
    # 4h EMA50 for trend direction
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d volume filter: 20-period average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # 1h volume-weighted average price (VWAP) deviation
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / vwap_den
    vwap_dev = (close - vwap) / vwap  # normalized deviation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(vwap_dev[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: only trade when 1h volume > 1.5x 1d average volume (scaled)
        vol_filter = volume[i] > 1.5 * vol_ma_20_1d_aligned[i] * (1/24)  # approximate 1h volume expectation
        
        if position == 0:
            # Long: price below VWAP (mean reversion), uptrend (above 4h EMA50), high volume
            if vwap_dev[i] < -0.002 and close[i] > ema_50_4h_aligned[i] and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: price above VWAP (mean reversion), downtrend (below 4h EMA50), high volume
            elif vwap_dev[i] > 0.002 and close[i] < ema_50_4h_aligned[i] and vol_filter:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses above VWAP or trend breaks
            if vwap_dev[i] > 0.001 or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses below VWAP or trend breaks
            if vwap_dev[i] < -0.001 or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals