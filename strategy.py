# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_1w_RSI_1D_Camarilla_Breakout
Hypothesis:
- Use weekly RSI(14) to determine market regime: RSI > 60 = bullish bias, RSI < 40 = bearish bias
- On 1d timeframe, calculate Camarilla pivot levels (R3, R4, S3, S4)
- In bullish regime (weekly RSI > 60): look for long breaks above R4 with volume confirmation
- In bearish regime (weekly RSI < 40): look for short breaks below S3 with volume confirmation
- Exit when price returns to the 1d VWAP or opposite Camarilla level is touched
- This combines weekly momentum bias with intraday mean reversion/breakout logic
- Designed to work in both bull (catch breakouts) and bear (fade at resistance) markets
- Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data once for RSI regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Load daily data for Camarilla levels and VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly RSI(14) for regime filter
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1w = np.where(rs == 100, 100, 100 - (100 / (1 + rs)))
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for VWAP
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    volume_1d = df_1d['volume'].values
    
    # VWAP calculation
    vwap_num = np.cumsum(typical_price_1d * volume_1d)
    vwap_den = np.cumsum(volume_1d)
    vwap_1d = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price_1d)
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    camarilla_r4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    camarilla_s4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align all indicators to 6h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Volume filter: 26-period average (1 day of 6h bars)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=26, min_periods=26).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            continue
        
        # Weekly RSI regime: >60 bullish, <40 bearish
        rsi_regime = rsi_1w_aligned[i]
        is_bullish = rsi_regime > 60
        is_bearish = rsi_regime < 40
        
        if position == 0:
            # Long entry: bullish regime + break above R4 + volume confirmation
            if (is_bullish and 
                close[i] > camarilla_r4_aligned[i] and 
                volume[i] > vol_ma[i] * 1.5):
                position = 1
                signals[i] = position_size
            # Short entry: bearish regime + break below S3 + volume confirmation
            elif (is_bearish and 
                  close[i] < camarilla_s3_aligned[i] and 
                  volume[i] > vol_ma[i] * 1.5):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Long exit: price returns to VWAP or touches S3 (mean reversion)
            if close[i] <= vwap_1d_aligned[i] or low[i] <= camarilla_s3_aligned[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Short exit: price returns to VWAP or touches R3 (mean reversion)
            if close[i] >= vwap_1d_aligned[i] or high[i] >= camarilla_r3_aligned[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_1w_RSI_1D_Camarilla_Breakout"
timeframe = "6h"
leverage = 1.0