#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA50 trend filter and volume confirmation.
- Uses 12h timeframe (primary) and 1d HTF for EMA50 trend alignment
- Camarilla levels calculated from prior 1d OHLC: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
- Breakout logic: long when price crosses above H3 with volume confirmation, short when price crosses below L3
- Trend filter: only long when price > 1d EMA50, only short when price < 1d EMA50
- Volume confirmation: current volume > 1.8 * 20-period volume MA to avoid low-volume false signals
- Exit: reverse signal or when price reverts to prior 12h VWAP (mean reversion)
- Discrete signal size: 0.25 to balance return and risk
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe as per research
- Works in both bull/bear: trend filter avoids counter-trend trades, Camarilla breakouts capture momentum in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate prior 12h VWAP for mean reversion exit
    typical_price = (high + low + close) / 3.0
    vwap_num = pd.Series(typical_price * volume).rolling(window=20, min_periods=20).sum().values
    vwap_den = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    vwap = vwap_num / np.where(vwap_den == 0, np.nan, vwap_den)
    vwap_aligned = vwap  # already on primary timeframe
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * volume_ma)
    
    # Trend filter: price above/below 1d EMA50
    uptrend = close > ema_50_1d_aligned
    downtrend = close < ema_50_1d_aligned
    
    # Calculate Camarilla H3/L3 from prior 1d data
    # Need to get prior 1d close, high, low for each 12h bar
    df_1d = get_htf_data(prices, '1d')  # already loaded above
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    camarilla_h3_1d = close_1d + 1.1 * (high_1d - low_1d) / 2.0
    camarilla_l3_1d = close_1d - 1.1 * (high_1d - low_1d) / 2.0
    
    # Align to 12h timeframe (completed 1d bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 1d EMA50 and VWAP/volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vwap_aligned[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above H3 AND uptrend AND volume confirmation
            if close[i] > camarilla_h3_aligned[i] and close[i-1] <= camarilla_h3_aligned[i-1] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below L3 AND downtrend AND volume confirmation
            elif close[i] < camarilla_l3_aligned[i] and close[i-1] >= camarilla_l3_aligned[i-1] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to VWAP (mean reversion) or reverse signal
            if close[i] <= vwap_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to VWAP (mean reversion) or reverse signal
            if close[i] >= vwap_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0