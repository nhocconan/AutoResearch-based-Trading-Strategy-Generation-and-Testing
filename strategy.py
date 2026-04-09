#!/usr/bin/env python3
# 4h_12h_camarilla_breakout_v3
# Hypothesis: Breakout above/below 12h Camarilla H3/L3 levels on 4h chart with volume confirmation.
# Long when price closes above H3 (bullish breakout), short when price closes below L3 (bearish breakout).
# Exit when price returns to opposite side of pivot point (mean reversion).
# Uses 12h trend filter: only take long trades when price > 12h EMA(50), only short trades when price < 12h EMA(50).
# Target: 15-35 trades/year (60-140 total over 4 years) with strict entry conditions.
# Works in both bull and bear markets due to breakout logic + trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_breakout_v3"
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
    
    # Load 12h data ONCE before loop for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for previous 12h period (using H3/L3 for tighter entries)
    ph = df_12h['high'].values  # previous period high
    pl = df_12h['low'].values   # previous period low
    pc = df_12h['close'].values # previous period close
    
    range_12h = ph - pl
    # H3 = close + (high - low) * 1.1/4
    # L3 = close - (high - low) * 1.1/4
    h3 = pc + range_12h * 1.1 / 4
    l3 = pc - range_12h * 1.1 / 4
    # Pivot point = (high + low + close) / 3
    pp = (ph + pl + pc) / 3
    
    # Align Camarilla levels to 4h timeframe (wait for previous 12h period's close)
    h3_aligned = align_htf_to_ltf(prices, df_12h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_12h, l3)
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp)
    
    # Load 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_12h = np.zeros_like(close_12h, dtype=float)
    ema_12h[0] = close_12h[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(close_12h)):
        ema_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_12h[i-1]
    
    # Align 12h EMA to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
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
        if np.isnan(atr[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely high volatility
        vol_filter = atr[i] < 0.06 * close[i]  # ATR less than 6% of price
        
        # Volume confirmation: current volume > 1.25x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.25
        
        # Trend filter: price > 12h EMA for longs, price < 12h EMA for shorts
        trend_long = close[i] > ema_12h_aligned[i]
        trend_short = close[i] < ema_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below pivot point (mean reversion)
            if close[i] < pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above pivot point (mean reversion)
            if close[i] > pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above H3 with volume confirmation, volatility filter, and trend filter
            if close[i] > h3_aligned[i] and vol_ok and vol_filter and trend_long:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below L3 with volume confirmation, volatility filter, and trend filter
            elif close[i] < l3_aligned[i] and vol_ok and vol_filter and trend_short:
                position = -1
                signals[i] = -0.25
    
    return signals