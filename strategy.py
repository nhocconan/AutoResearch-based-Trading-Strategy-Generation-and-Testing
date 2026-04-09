#!/usr/bin/env python3
# 4h_1d_donchian_breakout_v2
# Hypothesis: Breakout above/below Donchian(20) channel on 4h chart with volume confirmation.
# Uses 1d trend filter: only take long trades when price > 1d EMA(50), only short trades when price < 1d EMA(50).
# Exit when price returns to opposite side of Donchian midpoint (mean reversion).
# Target: 20-50 trades/year (80-200 total over 4 years) with strict entry conditions.
# Works in both bull and bear markets due to breakout logic + trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for volatility filter
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = 0.9 * atr[i-1] + 0.1 * tr[i]  # Wilder's smoothing
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = np.zeros_like(close_1d, dtype=float)
    ema_1d[0] = close_1d[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(close_1d)):
        ema_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_1d[i-1]
    
    # Align 1d EMA to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Donchian Channel (20 period) on 4h
    high_20 = np.zeros(n)
    low_20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            high_20[i] = np.nan
            low_20[i] = np.nan
        else:
            high_20[i] = np.max(high[i-19:i+1])
            low_20[i] = np.min(low[i-19:i+1])
    
    # Donchian midpoint
    midpoint = (high_20 + low_20) / 2
    
    # Volume confirmation - 20 period average
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(atr[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(midpoint[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely high volatility
        vol_filter = atr[i] < 0.05 * close[i]  # ATR less than 5% of price
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        # Trend filter: price > 1d EMA for longs, price < 1d EMA for shorts
        trend_long = close[i] > ema_1d_aligned[i]
        trend_short = close[i] < ema_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian midpoint (mean reversion)
            if close[i] < midpoint[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian midpoint (mean reversion)
            if close[i] > midpoint[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Enter long: price closes above Donchian upper band with volume confirmation, volatility filter, and trend filter
            if close[i] > high_20[i] and vol_ok and vol_filter and trend_long:
                position = 1
                signals[i] = 0.30
            # Enter short: price closes below Donchian lower band with volume confirmation, volatility filter, and trend filter
            elif close[i] < low_20[i] and vol_ok and vol_filter and trend_short:
                position = -1
                signals[i] = -0.30
    
    return signals