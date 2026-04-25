#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h for EMA50 trend filter.
- Williams %R(14): Oversold < -80 for long, overbought > -20 for short.
- Entry: Long when Williams %R crosses above -80 AND price > 12h EMA50 AND volume > 1.5 * 20-period average volume.
         Short when Williams %R crosses below -20 AND price < 12h EMA50 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Williams %R condition (long exits when %R crosses below -50, short exits when %R crosses above -50).
- Signal size: 0.25 discrete to minimize fee drag.
- Designed to capture mean reversals in trending markets while avoiding chop and low-volume false signals.
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # EMA50 on 12h close
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams %R(14) on 6h
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 6h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50)  # Need 20 for volume MA, 14 for Williams %R, 50 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_williams_r = williams_r[i]
        prev_williams_r = williams_r[i-1] if i > 0 else -50
        ema_50_level = ema_50_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20[i]
        
        # Williams %R crossover conditions
        crossed_above_80 = prev_williams_r <= -80 and curr_williams_r > -80
        crossed_below_20 = prev_williams_r >= -20 and curr_williams_r < -20
        crossed_above_50 = prev_williams_r <= -50 and curr_williams_r > -50
        crossed_below_50 = prev_williams_r >= -50 and curr_williams_r < -50
        
        # Trend filter: price relative to 12h EMA50
        price_above_ema = curr_close > ema_50_level
        price_below_ema = curr_close < ema_50_level
        
        # Exit conditions: opposite Williams %R crossover at midline
        if position != 0:
            # Exit long: Williams %R crosses below -50
            if position == 1:
                if crossed_below_50:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R crosses above -50
            elif position == -1:
                if crossed_above_50:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R reversal with volume and trend filters
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold reversal) AND price > EMA50 AND volume confirmation
            long_condition = crossed_above_80 and price_above_ema and volume_confirm
            
            # Short: Williams %R crosses below -20 (overbought reversal) AND price < EMA50 AND volume confirmation
            short_condition = crossed_below_20 and price_below_ema and volume_confirm
            
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

name = "6h_WilliamsR_Reversal_12hEMA50Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0