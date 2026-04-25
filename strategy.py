#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1w Supertrend trend filter and volume spike filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1w for Supertrend trend direction (more stable than daily) and Camarilla pivot levels from 1d.
- Camarilla Pivots: H3, L3 levels from prior 1d OHLC for breakout logic.
- Trend Filter: 1w Supertrend (ATR=10, mult=3.0) must align with breakout direction.
- Volume Filter: Current 4h volume > 2.0 * 20-period average 4h volume to confirm strong momentum.
- Entry: Long when close > H3 AND Supertrend uptrend AND volume spike.
         Short when close < L3 AND Supertrend downtrend AND volume spike.
- Exit: Opposite Camarilla break (long exits when close < L3, short exits when close > H3).
- Signal size: 0.25 discrete to minimize fee drag.
- Uses weekly Supertrend for stronger trend filtering that works in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivots (H3, L3) from prior day OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Prior day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # Shifted to avoid look-ahead
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla H3 and L3 levels (using standard Camarilla formula)
    camarilla_range = prev_high - prev_low
    h3 = prev_close + camarilla_range * 1.1 / 2
    l3 = prev_close - camarilla_range * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (waits for 1d bar close)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Calculate 1w Supertrend for trend filter (more stable than daily)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Supertrend calculation: ATR(10), multiplier=3.0
    atr_period = 10
    multiplier = 3.0
    
    # Calculate True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Calculate ATR
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate basic upper and lower bands
    hl2 = (high_1w + low_1w) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.full_like(close_1w, np.nan, dtype=float)
    direction = np.full_like(close_1w, np.nan, dtype=float)  # 1 for uptrend, -1 for downtrend
    
    # Set first valid value
    if not np.isnan(atr[atr_period-1]):
        supertrend[atr_period-1] = upper_band[atr_period-1]
        direction[atr_period-1] = 1  # Start with uptrend assumption
    
    # Calculate Supertrend iteratively
    for i in range(atr_period, len(close_1w)):
        if np.isnan(supertrend[i-1]) or np.isnan(direction[i-1]):
            continue
            
        close_val = close_1w[i]
        upper = upper_band[i]
        lower = lower_band[i]
        prev_st = supertrend[i-1]
        prev_dir = direction[i-1]
        
        if prev_dir == 1:  # Was in uptrend
            if close_val <= prev_st:
                # Reverse to downtrend
                supertrend[i] = upper
                direction[i] = -1
            else:
                # Continue uptrend
                supertrend[i] = max(prev_st, lower)
                direction[i] = 1
        else:  # Was in downtrend
            if close_val >= prev_st:
                # Reverse to uptrend
                supertrend[i] = lower
                direction[i] = 1
            else:
                # Continue downtrend
                supertrend[i] = min(prev_st, upper)
                direction[i] = -1
    
    # Align Supertrend direction to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1w, direction)
    
    # Calculate 4h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(atr_period, 20)  # Need ATR period for Supertrend, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(direction_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        h3_level = h3_aligned[i]
        l3_level = l3_aligned[i]
        supertrend_dir = direction_aligned[i]  # 1 for uptrend, -1 for downtrend
        
        # Volume spike: current volume > 2.0 * 20-period average volume
        volume_spike = curr_volume > 2.0 * vol_ma_20[i]
        
        # Camarilla breakout conditions
        broke_above_h3 = curr_close > h3_level
        broke_below_l3 = curr_close < l3_level
        
        # Trend alignment conditions from Supertrend
        uptrend = supertrend_dir == 1
        downtrend = supertrend_dir == -1
        
        # Exit conditions: opposite Camarilla break
        if position != 0:
            # Exit long: close breaks below L3
            if position == 1:
                if curr_close < l3_level:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close breaks above H3
            elif position == -1:
                if curr_close > h3_level:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend and volume filters
        if position == 0:
            # Long: break above H3 AND uptrend AND volume spike
            long_condition = broke_above_h3 and uptrend and volume_spike
            
            # Short: break below L3 AND downtrend AND volume spike
            short_condition = broke_below_l3 and downtrend and volume_spike
            
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

name = "4h_Camarilla_H3L3_Breakout_1wSupertrend_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0