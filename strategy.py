#!/usr/bin/env python3
# 12h_1w_donchian_breakout_v1
# Hypothesis: Breakout above/below 20-period Donchian channel on 12h chart with 1-week trend filter (EMA 50).
# Only take long when price > 1w EMA(50), short when price < 1w EMA(50).
# Exit when price crosses 1w EMA(50) in opposite direction.
# Uses volatility filter (ATR < 2.5% of price) and volume confirmation (volume > 1.5x 20-period avg).
# Target: 12-37 trades/year (50-150 total over 4 years) with strict entry conditions.
# Works in both bull and bear markets due to trend filter + volatility/volume filters reducing whipsaw.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_donchian_breakout_v1"
timeframe = "12h"
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
    
    # Donchian channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Load 1w data ONCE before loop for trend filter (EMA 50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA(50)
    close_1w = df_1w['close'].values
    ema_1w = np.zeros_like(close_1w, dtype=float)
    ema_1w[0] = close_1w[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(close_1w)):
        ema_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_1w[i-1]
    
    # Align 1w EMA to 12h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation - 20 period average
    vol_ma_20 = np.full(n, np.nan)
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
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(atr[i]) or np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely high volatility
        vol_filter = atr[i] < 0.025 * close[i]  # ATR less than 2.5% of price
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        # Trend filter: price > 1w EMA for longs, price < 1w EMA for shorts
        trend_long = close[i] > ema_1w_aligned[i]
        trend_short = close[i] < ema_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below 1w EMA
            if close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 1w EMA
            if close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume confirmation, volatility filter, and trend filter
            if close[i] > donchian_high[i] and vol_ok and vol_filter and trend_long:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume confirmation, volatility filter, and trend filter
            elif close[i] < donchian_low[i] and vol_ok and vol_filter and trend_short:
                position = -1
                signals[i] = -0.25
    
    return signals