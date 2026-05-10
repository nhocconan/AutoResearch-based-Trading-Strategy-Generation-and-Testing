#!/usr/bin/env python3
"""
1d_CCI_Trend_Filter_Volume
Hypothesis: CCI(20) on daily timeframe captures overbought/oversold conditions with mean reversion tendency.
In trending markets, CCI stays above/below zero for extended periods; in ranging markets, it oscillates around zero.
Combined with 1-week EMA50 trend filter and volume confirmation to avoid false signals.
Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
Target: 30-100 total trades over 4 years (7-25/year).
"""

name = "1d_CCI_Trend_Filter_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d CCI(20) calculation
    typical_price = (high + low + close) / 3.0
    sma_tp = np.full(n, np.nan)
    mad = np.full(n, np.nan)
    
    if n >= 20:
        # Calculate SMA of typical price
        sma_tp[19] = np.mean(typical_price[:20])
        for i in range(20, n):
            sma_tp[i] = (sma_tp[i-1] * 19 + typical_price[i]) / 20
        
        # Calculate Mean Absolute Deviation
        for i in range(19, n):
            tp_slice = typical_price[i-19:i+1]
            sma_tp_val = sma_tp[i]
            mad[i] = np.mean(np.abs(tp_slice - sma_tp_val))
        
        # CCI calculation
        cci = np.full(n, np.nan)
        for i in range(19, n):
            if mad[i] > 0:
                cci[i] = (typical_price[i] - sma_tp[i]) / (0.015 * mad[i])
            else:
                cci[i] = 0.0
    else:
        cci = np.full(n, np.nan)
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema50_1w[49] = np.mean(close_1w[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema50_1w[i-1]
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1d volume SMA20 for volume confirmation
    vol_sma20 = np.full(n, np.nan)
    if n >= 20:
        vol_sma20[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_sma20[i] = (vol_sma20[i-1] * 19 + volume[i]) / 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # warmup for CCI and EMA calculations
    
    for i in range(start_idx, n):
        if np.isnan(cci[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_sma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * vol_sma20[i]
        
        if position == 0:
            # Long: CCI oversold (< -100) in uptrend (price > weekly EMA50) with volume
            if cci[i] < -100 and close[i] > ema50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: CCI overbought (> 100) in downtrend (price < weekly EMA50) with volume
            elif cci[i] > 100 and close[i] < ema50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: CCI returns above -50 (mean reversion) or trend reversal
            if cci[i] > -50 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: CCI returns below 50 (mean reversion) or trend reversal
            if cci[i] < 50 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals