#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_Trend_ATRStop
Hypothesis: Donchian channel breakouts on 4h with volume confirmation and 1-day EMA trend filter capture
strong directional moves. ATR-based stop loss limits downside. Works in bull/bear by following
institutional flow with tight entry conditions to minimize fee drag.
Target: 20-50 trades/year (80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # 1-day EMA trend filter (34-period)
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_4h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # ATR for stop loss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 14)  # Warmup for Donchian and ATR
    
    for i in range(start_idx, n):
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(volume_filter[i]) or np.isnan(ema_1d_4h[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_20[i]
        lower = low_20[i]
        vol_ok = volume_filter[i]
        ema_trend = ema_1d_4h[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: break above upper Donchian with volume in uptrend
            if price > upper and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below lower Donchian with volume in downtrend
            elif price < lower and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long position: trail stop or reverse on breakdown
            stop_price = entry_price - 2.0 * atr_val
            if price <= stop_price or (price < ema_trend and price < upper):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short position: trail stop or reverse on breakout
            stop_price = entry_price + 2.0 * atr_val
            if price >= stop_price or (price > ema_trend and price > lower):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_Volume_Trend_ATRStop"
timeframe = "4h"
leverage = 1.0