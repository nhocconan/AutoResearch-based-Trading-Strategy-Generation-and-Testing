#!/usr/bin/env python3
# Strategy: 1h VWAP Pullback with 4h Trend & Volume Filter
# Hypothesis: In trending markets (4h EMA50), price pulls back to VWAP on 1h, offering high-probability entries.
# Volume spike confirms institutional interest. Works in both bull/bear by following 4h trend.
# Target: 15-30 trades/year via strict 4h trend + VWAP + volume confluence.

name = "1h_VWAP_Pullback_4hTrend_Volume"
timeframe = "1h"
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
    
    # 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA50 for trend direction
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h VWAP (volume-weighted average price)
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = vwap_numerator / vwap_denominator
    
    # 1h volume spike (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)  # 1.5x average volume
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Ensure 4h EMA50 is ready
    
    for i in range(start_idx, n):
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(vwap[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 4h uptrend, price pulls back to VWAP, volume spike
            if (close[i] > ema50_4h_aligned[i] and  # 4h uptrend
                close[i] <= vwap[i] * 1.005 and    # near/below VWAP (allow 0.5% overshoot)
                vol_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend, price pulls back to VWAP, volume spike
            elif (close[i] < ema50_4h_aligned[i] and  # 4h downtrend
                  close[i] >= vwap[i] * 0.995 and    # near/above VWAP (allow 0.5% overshoot)
                  vol_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: 4h trend reversal or price moves above VWAP + 0.5%
            if (close[i] < ema50_4h_aligned[i] or 
                close[i] > vwap[i] * 1.005):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: 4h trend reversal or price moves below VWAP - 0.5%
            if (close[i] > ema50_4h_aligned[i] or 
                close[i] < vwap[i] * 0.995):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals