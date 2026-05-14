#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d volume spike and ATR volatility filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for Camarilla pivot levels and volume spike filter.
- Camarilla Pivots: H3, L3 levels from prior 1d OHLC for breakout logic.
- Volume Filter: Current 4h volume > 1.8 * 20-period average 4h volume (avoid low-vol fakeouts).
- ATR Filter: Current ATR(14) < 1.8 * 20-period average ATR(14) to avoid extreme volatility whipsaws.
- Entry: Long when close > H3 AND volume confirmation AND ATR filter.
         Short when close < L3 AND volume confirmation AND ATR filter.
- Exit: Opposite Camarilla break (long exits when close < L3, short exits when close > H3).
- Signal size: 0.25 discrete to minimize fee drag.
- Designed to capture momentum bursts in both bull and bear markets while filtering chop/whipsaws.
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
    
    # Calculate 4h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) and its 20-period average for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0  # First bar has no previous close
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Need 20 for volume/ATR MA, 14 for ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or np.isnan(atr_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        h3_level = h3_aligned[i]
        l3_level = l3_aligned[i]
        
        # Volume confirmation: current volume > 1.8 * 20-period average volume
        volume_confirm = curr_volume > 1.8 * vol_ma_20[i]
        
        # ATR filter: current ATR < 1.8 * 20-period average ATR (avoid extreme volatility)
        atr_filter = curr_atr < 1.8 * atr_ma_20[i]
        
        # Camarilla breakout conditions
        broke_above_h3 = curr_close > h3_level
        broke_below_l3 = curr_close < l3_level
        
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
        
        # Entry conditions: Camarilla breakout with volume and ATR filters
        if position == 0:
            # Long: break above H3 AND volume confirmation AND ATR filter
            long_condition = broke_above_h3 and volume_confirm and atr_filter
            
            # Short: break below L3 AND volume confirmation AND ATR filter
            short_condition = broke_below_l3 and volume_confirm and atr_filter
            
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

name = "4h_Camarilla_H3L3_Breakout_1dVolSpike_ATRVolFilter_v1"
timeframe = "4h"
leverage = 1.0