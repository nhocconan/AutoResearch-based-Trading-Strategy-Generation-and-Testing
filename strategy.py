#!/usr/bin/env python3
# 4h_VolumeWeighted_Pivot_Pullback
# Hypothesis: Price pulling back to volume-weighted pivot points (VWAP) from prior 12h sessions during strong trends (12h EMA50) offers high-probability entries.
# Long when price > 12h EMA50 and pulls back to 12h VWAP support with volume confirmation; short when price < 12h EMA50 and pulls back to 12h VWAP resistance with volume confirmation.
# Uses volume-weighted average price (VWAP) as dynamic support/resistance and EMA for trend filter to work in both bull and bear markets.
# Designed for low trade frequency (target: 20-40 trades/year) with strict entry conditions.

name = "4h_VolumeWeighted_Pivot_Pullback"
timeframe = "4h"
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
    
    # Get 12h data for trend and VWAP
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h VWAP (volume-weighted average price)
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    vwap_numerator = (typical_price * df_12h['volume']).cumsum().values
    vwap_denominator = df_12h['volume'].cumsum().values
    vwap_12h = np.where(vwap_denominator > 0, vwap_numerator / vwap_denominator, 0)
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # Volume confirmation (20-period MA on 4h chart)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50) and VWAP (need cumulative volume, but align handles) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vwap_12h_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Price proximity to 12h VWAP (within 0.5% for pullback)
        vwap_dist_pct = abs(close[i] - vwap_12h_aligned[i]) / vwap_12h_aligned[i] * 100
        near_vwap = vwap_dist_pct < 0.5
        
        if position == 0:
            # Long entry: uptrend + pullback to VWAP support + volume spike
            if uptrend and near_vwap and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + pullback to VWAP resistance + volume spike
            elif downtrend and near_vwap and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks down or price moves significantly above VWAP
            if not uptrend or vwap_dist_pct > 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks up or price moves significantly below VWAP
            if not downtrend or vwap_dist_pct > 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals