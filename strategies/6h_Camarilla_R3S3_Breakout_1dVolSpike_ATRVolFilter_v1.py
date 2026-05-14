#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d volume spike and ATR volatility filter.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Camarilla pivot levels and volume spike filter.
- Camarilla Pivots: R3, S3 levels from prior 1d OHLC for breakout/fade logic.
- Volume Filter: Current 6h volume > 2.0 * 20-period average 6h volume (avoid low-vol fakeouts).
- ATR Filter: Current ATR(14) < 2.0 * 20-period average ATR(14) to avoid extreme volatility whipsaws.
- Entry: Long when close > R3 AND volume confirmation AND ATR filter.
         Short when close < S3 AND volume confirmation AND ATR filter.
- Exit: Opposite Camarilla break (long exits when close < S3, short exits when close > R3).
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
    
    # Calculate 1d Camarilla pivots (R3, S3) from prior day OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Prior day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # Shifted to avoid look-ahead
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R3 and S3 levels
    camarilla_range = prev_high - prev_low
    r3 = prev_close + camarilla_range * 1.1 / 4
    s3 = prev_close - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe (waits for 1d bar close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 6h volume average for confirmation (20-period)
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
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or np.isnan(atr_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20[i]
        
        # ATR filter: current ATR < 2.0 * 20-period average ATR (avoid extreme volatility)
        atr_filter = curr_atr < 2.0 * atr_ma_20[i]
        
        # Camarilla breakout conditions
        broke_above_r3 = curr_close > r3_level
        broke_below_s3 = curr_close < s3_level
        
        # Exit conditions: opposite Camarilla break
        if position != 0:
            # Exit long: close breaks below S3
            if position == 1:
                if curr_close < s3_level:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close breaks above R3
            elif position == -1:
                if curr_close > r3_level:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume and ATR filters
        if position == 0:
            # Long: break above R3 AND volume confirmation AND ATR filter
            long_condition = broke_above_r3 and volume_confirm and atr_filter
            
            # Short: break below S3 AND volume confirmation AND ATR filter
            short_condition = broke_below_s3 and volume_confirm and atr_filter
            
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

name = "6h_Camarilla_R3S3_Breakout_1dVolSpike_ATRVolFilter_v1"
timeframe = "6h"
leverage = 1.0