#!/usr/bin/env python3
"""
4h Volume Spike + Close Above/Below Prior Day VWAP with 1D EMA Trend Filter
Long: Close > prior day VWAP + volume > 2x 4h volume MA(20) + close > 1D EMA50
Short: Close < prior day VWAP + volume > 2x 4h volume MA(20) + close < 1D EMA50
Exit: Close crosses back below/above prior day VWAP
Uses VWAP for intraday mean reversion edge and volume surge for momentum confirmation
Target: 25-35 trades/year per symbol
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
    
    # Get 1D data for VWAP and trend filter
    df_1d = get_htf_data(prices, '1d')
    # Calculate VWAP for each 1D bar: cumulative (price * volume) / cumulative volume
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    # Shift by 1 to get prior day's VWAP (not current forming day)
    prior_vwap = vwap.shift(1)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    prior_vwap_aligned = align_htf_to_ltf(prices, df_1d, prior_vwap.values)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h volume moving average (20-period for confirmation)
    df_4h = get_htf_data(prices, '4h')
    volume_ma_20 = pd.Series(df_4h['volume']).rolling(window=20, min_periods=20).mean()
    volume_ma_20_4h = align_htf_to_ltf(prices, df_4h, volume_ma_20.values)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(prior_vwap_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20_4h[i]
        
        if position == 0:
            # Long: close above prior day VWAP + volume spike + 1D uptrend
            if price > prior_vwap_aligned[i] and vol > 2.0 * vol_ma and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: close below prior day VWAP + volume spike + 1D downtrend
            elif price < prior_vwap_aligned[i] and vol > 2.0 * vol_ma and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: close back below prior day VWAP (mean reversion)
            if price < prior_vwap_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close back above prior day VWAP (mean reversion)
            if price > prior_vwap_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VolumeSpike_PriorVWAP_1DEMA50"
timeframe = "4h"
leverage = 1.0