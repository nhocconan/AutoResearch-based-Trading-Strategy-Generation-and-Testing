#!/usr/bin/env python3
"""
6h_Keltner_Breakout_ATR_Volume_Filter
Hypothesis: Keltner Channel breakouts combined with ATR-based volatility filter and volume confirmation.
In bull markets, price breaks above upper Keltner band in uptrend; in bear markets, breaks below lower band in downtrend.
ATR filter ensures sufficient volatility for meaningful moves, volume confirms institutional participation.
Uses 12h EMA50 as higher timeframe trend filter to avoid counter-trend trades.
Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
"""

name = "6h_Keltner_Breakout_ATR_Volume_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(20) for Keltner channels and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Keltner Channel parameters
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_mult = 2.0
    upper_keltner = ema_20 + (keltner_mult * atr)
    lower_keltner = ema_20 - (keltner_mult * atr)
    
    # Volume confirmation (24-period MA on 6h = ~6 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA20 (20), ATR (20), volume MA (24), 12h EMA50 (50)
    start_idx = max(20, 20, 24, 50)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(upper_keltner[i]) or 
            np.isnan(lower_keltner[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher timeframe trend filter
        uptrend_12h = close[i] > ema_50_12h_aligned[i]
        downtrend_12h = close[i] < ema_50_12h_aligned[i]
        
        # Volatility filter: ATR > 30-period MA of ATR (ensures sufficient volatility)
        atr_ma = pd.Series(atr).rolling(window=30, min_periods=30).mean().values
        vol_filter = atr[i] > atr_ma[i] if not np.isnan(atr_ma[i]) else False
        
        # Volume confirmation (>1.8x average volume)
        volume_confirm = volume[i] > volume_ma[i] * 1.8
        
        if position == 0:
            # Long entry: uptrend + price breaks above upper Keltner + volatility + volume
            if uptrend_12h and close[i] > upper_keltner[i] and vol_filter and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price breaks below lower Keltner + volatility + volume
            elif downtrend_12h and close[i] < lower_keltner[i] and vol_filter and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below upper Keltner
            if not uptrend_12h or close[i] < ema_20[i]:  # Exit to EMA20 (middle of channel)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters above lower Keltner
            if not downtrend_12h or close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals