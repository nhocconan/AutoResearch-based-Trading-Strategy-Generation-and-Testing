#!/usr/bin/env python3
# 6h_Pivot_Reversion_With_Trend
# Hypothesis: In both bull and bear markets, price often reverts to the 12-hour VWAP after deviating,
# especially when the 1-day trend (via EMA50) is strong. We enter reversals when price deviates
# significantly from 12h VWAP with volume confirmation, in the direction of the 1-day trend.
# Uses 12h VWAP for mean reversion and 1d EMA50 for trend filter. Designed for low trade frequency.

name = "6h_Pivot_Reversion_With_Trend"
timeframe = "6h"
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
    
    # Get 12h data for VWAP (mean reversion target)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h VWAP: cumulative (price * volume) / cumulative volume
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    pv_12h = typical_price_12h * volume_12h
    cum_pv_12h = np.cumsum(pv_12h)
    cum_vol_12h = np.cumsum(volume_12h)
    vwap_12h = np.divide(cum_pv_12h, cum_vol_12h, out=np.full_like(cum_pv_12h, np.nan), where=cum_vol_12h!=0)
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (20-period average on 6h)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    # Deviation from 12h VWAP as percentage
    deviation = np.zeros(n)
    for i in range(n):
        if not np.isnan(vwap_12h_aligned[i]) and vwap_12h_aligned[i] != 0:
            deviation[i] = (close[i] - vwap_12h_aligned[i]) / vwap_12h_aligned[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20) + 5  # need enough history
    
    for i in range(start_idx, n):
        if np.isnan(vwap_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price below VWAP (oversold), volume spike, and 1d trend up (price > EMA50)
            if deviation[i] < -0.015 and volume_confirm and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price above VWAP (overbought), volume spike, and 1d trend down (price < EMA50)
            elif deviation[i] > 0.015 and volume_confirm and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back above VWAP or trend weakens
            if deviation[i] > 0.005 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back below VWAP or trend weakens
            if deviation[i] < -0.005 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals