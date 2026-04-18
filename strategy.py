#!/usr/bin/env python3
"""
1d_Weekly_KAMA_Direction_1wTrend_Filter
Hypothesis: Weekly KAMA direction (1w) filters daily price action to capture institutional moves.
In bull markets, price follows weekly KAMA up; in bear markets, price follows weekly KAMA down.
Daily mean reversion at Bollinger Bands (20,2) provides entries in direction of weekly trend.
Volume confirmation filters low-quality signals. Target: 10-25 trades/year (40-100 total over 4 years).
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
    
    # Weekly data for KAMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly KAMA (adaptive moving average)
    close_1w = df_1w['close']
    change_1w = abs(close_1w.diff(1))
    vol_1w = abs(close_1w.diff(10)).rolling(window=10, min_periods=10).sum()
    er_1w = change_1w / vol_1w.replace(0, 1e-10)
    sc_1w = (er_1w * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama_1w = [close_1w.iloc[0]]
    for i in range(1, len(close_1w)):
        kama_1w.append(kama_1w[-1] + sc_1w.iloc[i] * (close_1w.iloc[i] - kama_1w[-1]))
    kama_1w = np.array(kama_1w)
    
    # Align weekly KAMA to daily (waits for weekly bar to close)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Daily Bollinger Bands (20,2)
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean()
    dev = 2 * close_s.rolling(window=20, min_periods=20).std()
    upper = basis + dev
    lower = basis - dev
    
    # Volume filter: >1.3x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40  # Warmup for BB and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(basis[i]) or
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_1w_aligned[i]
        bb_lower = lower[i]
        bb_upper = upper[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: price at lower BB in weekly uptrend with volume
            if price <= bb_lower and price > kama_val and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price at upper BB in weekly downtrend with volume
            elif price >= bb_upper and price < kama_val and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses weekly KAMA or returns to BB middle
            if price >= kama_val or price >= basis[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses weekly KAMA or returns to BB middle
            if price <= kama_val or price <= basis[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_KAMA_Direction_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0