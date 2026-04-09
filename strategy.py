#!/usr/bin/env python3
# 12h_1d_camarilla_breakout_v2
# Hypothesis: Breakout above/below 1d Camarilla pivot levels (H4/L4) on 12h chart with volume confirmation.
# Long when price closes above H4 (bullish breakout), short when price closes below L4 (bearish breakout).
# Exit when price returns to pivot point (mean reversion) or volatility filter fails.
# Works in bull markets by capturing breakouts, in bear markets by fading false breakouts at key levels.
# Target: 15-30 trades/year (60-120 total over 4 years) with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
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
    
    # Load 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for previous day
    ph = df_1d['high'].values  # previous day high
    pl = df_1d['low'].values   # previous day low
    pc = df_1d['close'].values # previous day close
    
    range_1d = ph - pl
    # H4 = close + 1.5 * (high - low) * 1.1/2
    # L4 = close - 1.5 * (high - low) * 1.1/2
    h4 = pc + 1.5 * range_1d * 1.1 / 2
    l4 = pc - 1.5 * range_1d * 1.1 / 2
    # Pivot point = (high + low + close) / 3
    pp = (ph + pl + pc) / 3
    
    # Align Camarilla levels to 12h timeframe (wait for previous day's close)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
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
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(atr[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely high volatility
        vol_filter = atr[i] < 0.06 * close[i]  # ATR less than 6% of price
        
        # Volume confirmation: current volume > 1.25x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.25
        
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
            # Enter long: price closes above H4 with volume confirmation and volatility filter
            if close[i] > h4_aligned[i] and vol_ok and vol_filter:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below L4 with volume confirmation and volatility filter
            elif close[i] < l4_aligned[i] and vol_ok and vol_filter:
                position = -1
                signals[i] = -0.25
    
    return signals