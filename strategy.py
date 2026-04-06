#!/usr/bin/env python3
"""
6h VWAP + Volume Spike + 12h Trend Filter
Hypothesis: On 6h timeframe, price returning to VWAP after a volume spike, aligned with 12h trend,
provides high-probability entries in both bull and bear markets. VWAP acts as dynamic support/resistance,
volume spikes indicate institutional interest, and 12h trend filter avoids counter-trend trades.
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_vwap_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # VWAP calculation (typical price * volume)
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Volume filter: current volume > 2.0x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 30-period EMA on 12h for trend (slower to reduce whipsaw)
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 30:
        ema_12h[29] = np.mean(close_12h[:30])
        for i in range(30, len(close_12h)):
            ema_12h[i] = (close_12h[i] * 2 + ema_12h[i-1] * 28) / 30  # alpha = 2/30
    
    # Trend: 1 if close > EMA (uptrend), -1 if close < EMA (downtrend)
    trend_12h = np.where(close_12h > ema_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(30, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(vwap[i]) or np.isnan(vol_ma[i]) or np.isnan(trend_12h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: significant volume spike
        volume_spike = volume[i] > vol_ma[i] * 2.0
        
        # Price proximity to VWAP: within 0.5% of VWAP
        price_vwap_ratio = close[i] / vwap[i]
        near_vwap = (price_vwap_ratio >= 0.995) & (price_vwap_ratio <= 1.005)
        
        # Check exits and stoploss (using 2.0 * price range as proxy for volatility)
        if position == 1:  # long position
            # Exit: price moves below VWAP OR trend turns down
            if (close[i] < vwap[i] or trend_12h_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price moves above VWAP OR trend turns up
            if (close[i] > vwap[i] or trend_12h_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for mean-reversion entries near VWAP with volume spike
            # Long: price at VWAP support in uptrend with volume spike
            if (near_vwap and trend_12h_aligned[i] == 1 and volume_spike):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price at VWAP resistance in downtrend with volume spike
            elif (near_vwap and trend_12h_aligned[i] == -1 and volume_spike):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals