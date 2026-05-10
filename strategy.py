#!/usr/bin/env python3
# 4h_CCI_Trend_Follow_20_50_1dTrend_Volume
# Hypothesis: CCI (20) identifies overbought/oversold conditions. In a daily uptrend, buy when CCI crosses above -100 from below (end of pullback). In a daily downtrend, sell when CCI crosses below 100 from above (end of bounce). Volume confirmation filters weak moves. Designed for ~25-40 trades/year to avoid fee drag.

name = "4h_CCI_Trend_Follow_20_50_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # CCI (20)
    typical_price = (high + low + close) / 3.0
    tp_series = pd.Series(typical_price)
    ma_tp = tp_series.rolling(window=20, min_periods=20).mean()
    mad = tp_series.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp_series - ma_tp) / (0.015 * mad)
    cci = cci.values
    
    # Daily trend: EMA50 on daily close
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align daily trend to 4h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volume confirmation: 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(cci[i]) or np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: CCI crosses above -100 in daily uptrend with volume
            if i > start_idx and cci[i-1] <= -100 and cci[i] > -100 and \
               trend_1d_up_aligned[i] > 0.5 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: CCI crosses below 100 in daily downtrend with volume
            elif i > start_idx and cci[i-1] >= 100 and cci[i] < 100 and \
                 trend_1d_down_aligned[i] > 0.5 and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when CCI crosses below +100 (overbought) or trend fails
            if (cci[i] < 100 and cci[i-1] >= 100) or trend_1d_up_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when CCI crosses above -100 (oversold) or trend fails
            if (cci[i] > -100 and cci[i-1] <= -100) or trend_1d_down_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals