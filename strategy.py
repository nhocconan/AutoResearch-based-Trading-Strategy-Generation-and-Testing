#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla Pivot (H3/L3) breakout with 1d EMA34 trend filter and volume confirmation.
- Camarilla levels from 1d: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
- Long when price breaks above H3 with volume > 1.5 * volume SMA20 AND price > 1d EMA34 (uptrend)
- Short when price breaks below L3 with volume > 1.5 * volume SMA20 AND price < 1d EMA34 (downtrend)
- Exit when price retreats to pivot point (PP) or opposite Camarilla level (L3 for long, H3 for short)
- Designed to capture institutional breakouts with trend alignment and volume confirmation
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d OHLC for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough data for EMA34
        return np.zeros(n)
    
    # Camarilla levels from 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP) = (high + low + close) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla H3/L3 = close ± 1.1*(high-low)/4
    h3_1d = close_1d + 1.1 * (high_1d - low_1d) / 4.0
    l3_1d = close_1d - 1.1 * (high_1d - low_1d) / 4.0
    
    # Align HTF levels to LTF (12h)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5 * volume SMA20
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * volume_sma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need EMA34 and volume SMA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(h3_1d_aligned[i]) or 
            np.isnan(l3_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above H3 with volume confirmation AND uptrend
            if close[i] > h3_1d_aligned[i] and volume_filter[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 with volume confirmation AND downtrend
            elif close[i] < l3_1d_aligned[i] and volume_filter[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price retreats to pivot point (PP) or below L3
            if close[i] <= pp_1d_aligned[i] or close[i] < l3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price retreats to pivot point (PP) or above H3
            if close[i] >= pp_1d_aligned[i] or close[i] > h3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0