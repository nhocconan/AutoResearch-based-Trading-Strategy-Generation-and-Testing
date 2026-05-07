#!/usr/bin/env python3
# 6h_Liquidity_Imbalance_Fade
# Hypothesis: Large price gaps create liquidity imbalances that get filled by mean-reversion.
# Fade gaps against the 1-week trend using volume confirmation.
# Uses weekly trend filter (EMA50 on weekly close) to determine bias.
# Targets 10-25 trades/year with position size 0.25 to avoid fee drag.
# Works in both bull/bear: fades gaps against weekly trend (mean reversion in trends).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Liquidity_Imbalance_Fade"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = weekly_ema50 > 0  # Will be replaced with actual comparison
    weekly_uptrend = weekly_close > weekly_ema50
    
    # Load daily data ONCE for gap detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily gap detection: today's open vs yesterday's close
    daily_open = df_1d['open'].values
    daily_close = df_1d['close'].values
    gap_up = (daily_open > daily_close).astype(float)  # Gap up when open > prior close
    gap_down = (daily_open < daily_close).astype(float)  # Gap down when open < prior close
    
    # Align weekly trend and daily gaps to 6h timeframe
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    gap_up_aligned = align_htf_to_ltf(prices, df_1d, gap_up)
    gap_down_aligned = align_htf_to_ltf(prices, df_1d, gap_down)
    
    # Volume spike detection: current volume > 2.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = np.where(vol_ma > 0, volume / vol_ma, 1.0) > 2.5
    
    # Mean reversion signal: price deviation from 24-period VWAP
    typical_price = (high + low + close) / 3.0
    vwap_num = pd.Series(typical_price * volume).rolling(window=24, min_periods=24).sum().values
    vwap_den = pd.Series(volume).rolling(window=24, min_periods=24).sum().values
    vwap = np.where(vwap_den > 0, vwap_num / vwap_den, typical_price)
    price_dev = (close - vwap) / vwap  # Normalized deviation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_uptrend_aligned[i]) or np.isnan(gap_up_aligned[i]) or 
            np.isnan(gap_down_aligned[i]) or np.isnan(vol_spike[i]) or
            np.isnan(price_dev[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: gap down during weekly uptrend + volume spike + price below VWAP
            long_condition = (gap_down_aligned[i] > 0.5) and weekly_uptrend_aligned[i] and vol_spike[i] and (price_dev[i] < -0.015)
            # Short: gap up during weekly downtrend + volume spike + price above VWAP
            short_condition = (gap_up_aligned[i] > 0.5) and (not weekly_uptrend_aligned[i]) and vol_spike[i] and (price_dev[i] > 0.015)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: gap up or price reverts to VWAP
            if (gap_up_aligned[i] > 0.5) or (price_dev[i] > -0.005):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: gap down or price reverts to VWAP
            if (gap_down_aligned[i] > 0.5) or (price_dev[i] < 0.005):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals