#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H4/L4 breakout with 1w EMA50 trend filter and 1d volume spike filter.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for EMA50 trend direction (filters against weekly trend).
- HTF: 1d for Camarilla pivot levels (H4/L4) and volume confirmation.
- Camarilla Pivots: H4, L4 levels from prior 1d OHLC for breakout logic (stronger levels than H3/L3).
- Trend Filter: 1w EMA50 must align with breakout direction (long: close > EMA50, short: close < EMA50).
- Volume Filter: Current 1d volume > 2.0 * 20-period average 1d volume (using daily data for cleaner signal).
- Entry: Long when close > H4 AND close > 1w EMA50 AND 1d volume spike.
         Short when close < L4 AND close < 1w EMA50 AND 1d volume spike.
- Exit: Opposite Camarilla break (long exits when close < L4, short exits when close > H4).
- Signal size: 0.25 discrete to minimize fee drag.
- Designed to capture strong momentum bursts aligned with weekly trend while filtering chop/whipsaws.
- Uses weekly trend filter to avoid counter-trend trades in bear markets (2022 crash, 2025 range).
- Uses daily Camarilla and volume for cleaner signals less prone to 6h noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Camarilla pivots (H4, L4) from prior day OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Prior day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # Shifted to avoid look-ahead
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla H4 and L4 levels (using standard Camarilla formula)
    camarilla_range = prev_high - prev_low
    h4 = prev_close + camarilla_range * 1.1
    l4 = prev_close - camarilla_range * 1.1
    
    # Align Camarilla levels to 6h timeframe (waits for 1d bar close)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate 1d volume average for confirmation (20-period)
    df_1d_vol = get_htf_data(prices, '1d')
    if len(df_1d_vol) < 1:
        return np.zeros(n)
    vol_1d = df_1d_vol['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d_vol, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for weekly EMA, 20 for daily volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        h4_level = h4_aligned[i]
        l4_level = l4_aligned[i]
        ema_50_level = ema_50_1w_aligned[i]
        
        # Exit conditions: opposite Camarilla break
        if position != 0:
            # Exit long: close breaks below L4
            if position == 1:
                if curr_close < l4_level:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close breaks above H4
            elif position == -1:
                if curr_close > h4_level:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend and volume filters
        if position == 0:
            # Long: break above H4 AND above weekly EMA50
            long_condition = (curr_close > h4_level) and (curr_close > ema_50_level)
            
            # Short: break below L4 AND below weekly EMA50
            short_condition = (curr_close < l4_level) and (curr_close < ema_50_level)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H4L4_Breakout_1wEMA50_Trend_v1"
timeframe = "6h"
leverage = 1.0