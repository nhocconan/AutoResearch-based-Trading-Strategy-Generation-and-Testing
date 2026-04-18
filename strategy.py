#!/usr/bin/env python3
"""
12h_VWAP_RangeReversion_BollingerBands
Hypothesis: 12-hour VWAP mean reversion with Bollinger Bands and volume confirmation.
Trades mean reversion from Bollinger Band extremes toward VWAP, using 1-day trend filter
to avoid counter-trend trades. Designed for low frequency (15-35 trades/year) with
strong performance in ranging markets by combining intraday mean reversion with
daily trend alignment.
"""

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
    
    # Get 12h data for VWAP calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h VWAP
    typical_price_12h = (df_12h['high'].values + df_12h['low'].values + df_12h['close'].values) / 3
    vwap_num = np.cumsum(typical_price_12h * df_12h['volume'].values)
    vwap_den = np.cumsum(df_12h['volume'].values)
    vwap_12h = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day EMA50 trend filter
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[0:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = close_1d[i] * alpha + ema50_1d[i-1] * (1 - alpha)
    
    # Calculate Bollinger Bands (20, 2) on 12h typical price
    tp_12h = typical_price_12h
    sma_20 = np.full(len(tp_12h), np.nan)
    std_20 = np.full(len(tp_12h), np.nan)
    
    for i in range(19, len(tp_12h)):
        sma_20[i] = np.mean(tp_12h[i-19:i+1])
        std_20[i] = np.std(tp_12h[i-19:i+1])
    
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Align all indicators to 12h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    upper_bb_aligned = align_htf_to_ltf(prices, df_12h, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_12h, lower_bb)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(vwap_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price touches lower BB with volume confirmation and above 1d EMA50
            if (close[i] <= lower_bb_aligned[i] and vol_confirm[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches upper BB with volume confirmation and below 1d EMA50
            elif (close[i] >= upper_bb_aligned[i] and vol_confirm[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches VWAP or breaks below lower BB
            if (close[i] >= vwap_aligned[i] or close[i] < lower_bb_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches VWAP or breaks above upper BB
            if (close[i] <= vwap_aligned[i] or close[i] > upper_bb_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_VWAP_RangeReversion_BollingerBands"
timeframe = "12h"
leverage = 1.0