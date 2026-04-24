#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA50 trend direction and volume spike confirmation.
- Camarilla Pivot: Calculated from prior day's high/low/close (using 1d data).
- R3/S3 levels: Aggressive breakout levels for strong momentum.
- Trend Filter: Only trade breakouts in direction of 4h EMA50 (avoid counter-trend whipsaws).
- Volume Confirmation: Current 1h volume > 1.5 * 20-period average volume (using 4h volume data aligned).
- Session Filter: Trade only during 08-20 UTC (avoid low-volume Asian session).
- Signal Size: 0.20 discrete to minimize fee drag.
- Works in bull markets via long breakouts, bear markets via short breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla pivot calculation (prior day HLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d bar
    # R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use prior day's HLC (no look-ahead)
    h_shift = np.concatenate([[np.nan], high_1d[:-1]])
    l_shift = np.concatenate([[np.nan], low_1d[:-1]])
    c_shift = np.concatenate([[np.nan], close_1d[:-1]])
    
    camarilla_range = h_shift - l_shift
    r3 = c_shift + (camarilla_range * 1.1 / 4)
    s3 = c_shift - (camarilla_range * 1.1 / 4)
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Get 4h data for EMA50 trend and volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4h EMA50 for trend direction
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 4h 20-period average volume for confirmation
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready (check for NaN)
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma_20_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: only trade in direction of 4h EMA50
        # Long bias when price > EMA50, short bias when price < EMA50
        long_bias = curr_close > ema50_4h_aligned[i]
        short_bias = curr_close < ema50_4h_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_4h_aligned[i]
        
        # Exit conditions: price returns to Camarilla pivot levels
        if position != 0:
            # Exit long: price < S3 (support level)
            if position == 1:
                if curr_close < s3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > R3 (resistance level)
            elif position == -1:
                if curr_close > r3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend and volume filters
        if position == 0:
            # Long: price > R3 AND long bias AND volume confirmation
            long_condition = (curr_close > r3_aligned[i] and 
                            long_bias and
                            volume_confirm)
            
            # Short: price < S3 AND short bias AND volume confirmation
            short_condition = (curr_close < s3_aligned[i] and 
                             short_bias and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hEMA50Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0