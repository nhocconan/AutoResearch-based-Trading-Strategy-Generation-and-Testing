#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 1d volume spike filter and weekly trend filter.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Camarilla pivot levels and volume spike confirmation (>2x 20-period average).
- Weekly trend: Price > weekly EMA20 for long bias, Price < weekly EMA20 for short bias.
- Entry: Long when close breaks above Camarilla R3 AND weekly uptrend AND volume spike.
         Short when close breaks below Camarilla S3 AND weekly downtrend AND volume spike.
- Exit: Opposite Camarilla level (long exits at S3, short exits at R3) or weekly trend reversal.
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla pivots work well in ranging markets (common in 2025 BTC/ETH) while volume spike confirms institutional interest.
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
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivots (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for previous day calculation
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate weekly EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 2)  # Need 20 for volume MA, 2 for Camarilla (shifted)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(ema20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Weekly trend filter
        weekly_uptrend = curr_close > ema20_1w_aligned[i]
        weekly_downtrend = curr_close < ema20_1w_aligned[i]
        
        # Camarilla breakout conditions
        broke_above_r3 = curr_close > camarilla_r3_aligned[i]
        broke_below_s3 = curr_close < camarilla_s3_aligned[i]
        
        # Exit conditions
        if position != 0:
            # Exit long: price breaks below S3 or weekly trend turns down
            if position == 1:
                if curr_close < camarilla_s3_aligned[i] or not weekly_uptrend:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above R3 or weekly trend turns up
            elif position == -1:
                if curr_close > camarilla_r3_aligned[i] or not weekly_downtrend:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and weekly trend
        if position == 0:
            # Long: break above R3 AND weekly uptrend AND volume confirmation
            long_condition = broke_above_r3 and weekly_uptrend and volume_confirm
            
            # Short: break below S3 AND weekly downtrend AND volume confirmation
            short_condition = broke_below_s3 and weekly_downtrend and volume_confirm
            
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

name = "6h_Camarilla_R3S3_Breakout_1dVolumeSpike_1wEMA20Trend_v1"
timeframe = "6h"
leverage = 1.0