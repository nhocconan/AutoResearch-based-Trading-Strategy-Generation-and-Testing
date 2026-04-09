#!/usr/bin/env python3
# 1d_1w_camarilla_breakout_v1
# Hypothesis: Breakout above/below weekly Camarilla H4/L4 levels on daily chart with volume confirmation.
# Uses weekly trend filter: only take long trades when price > weekly EMA(20), only short trades when price < weekly EMA(20).
# Exit when price returns to weekly pivot point (mean reversion).
# Target: 10-25 trades/year (40-100 total over 4 years) with strict entry conditions.
# Works in both bull and bear markets due to breakout logic + trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
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
    
    # Load weekly data ONCE before loop for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for previous week (using H4/L4 for tighter entries)
    ph = df_1w['high'].values  # previous week high
    pl = df_1w['low'].values   # previous week low
    pc = df_1w['close'].values # previous week close
    
    range_1w = ph - pl
    # H4 = close + (high - low) * 1.1/2
    # L4 = close - (high - low) * 1.1/2
    h4 = pc + range_1w * 1.1 / 2
    l4 = pc - range_1w * 1.1 / 2
    # Pivot point = (high + low + close) / 3
    pp = (ph + pl + pc) / 3
    
    # Align Camarilla levels to daily timeframe (wait for previous week's close)
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    
    # Load weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = np.zeros_like(close_1w, dtype=float)
    ema_1w[0] = close_1w[0]
    alpha = 2.0 / (20 + 1)
    for i in range(1, len(close_1w)):
        ema_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_1w[i-1]
    
    # Align weekly EMA to daily timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
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
        if np.isnan(atr[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely high volatility
        vol_filter = atr[i] < 0.05 * close[i]  # ATR less than 5% of price (tighter)
        
        # Volume confirmation: current volume > 1.5x 20-period average (tighter)
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        # Trend filter: price > weekly EMA for longs, price < weekly EMA for shorts
        trend_long = close[i] > ema_1w_aligned[i]
        trend_short = close[i] < ema_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below weekly pivot point (mean reversion)
            if close[i] < pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly pivot point (mean reversion)
            if close[i] > pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above H4 with volume confirmation, volatility filter, and trend filter
            if close[i] > h4_aligned[i] and vol_ok and vol_filter and trend_long:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below L4 with volume confirmation, volatility filter, and trend filter
            elif close[i] < l4_aligned[i] and vol_ok and vol_filter and trend_short:
                position = -1
                signals[i] = -0.25
    
    return signals