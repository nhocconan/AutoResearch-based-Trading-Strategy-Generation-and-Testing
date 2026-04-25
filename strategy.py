#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA50 trend direction and volume spike filter.
- Williams %R(14): Oversold < -80 for long, Overbought > -20 for short.
- Trend Filter: Only take longs when price > 1d EMA50, shorts when price < 1d EMA50.
- Volume Confirmation: Current 6h volume > 1.5 * 20-period average 6h volume.
- Entry: Long when %R crosses above -80 AND trend up AND volume confirmation.
         Short when %R crosses below -20 AND trend down AND volume confirmation.
- Exit: Opposite %R crossover (%R crosses below -50 for long exit, above -50 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Designed to capture mean reversals in trending markets, working in both bull and bear phases.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    vol_ma_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Williams %R(14) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r[highest_high == lowest_low] = -50  # neutral value
    
    # Calculate 6h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50)  # Need 20 for volume MA, 14 for Williams %R, 50 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(williams_r[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_williams_r = williams_r[i]
        prev_williams_r = williams_r[i-1] if i > 0 else -50
        ema_50_level = ema_50_aligned[i]
        vol_ma_20_1d_level = vol_ma_20_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume (both 6h and 1d)
        volume_confirm_6h = curr_volume > 1.5 * vol_ma_20[i]
        volume_confirm_1d = volume[i] > 1.5 * vol_ma_20_1d_level  # approximate 6h volume vs 1d MA
        volume_confirm = volume_confirm_6h and volume_confirm_1d
        
        # Williams %R crossover conditions
        crossed_above_80 = prev_williams_r <= -80 and curr_williams_r > -80
        crossed_below_20 = prev_williams_r >= -20 and curr_williams_r < -20
        crossed_below_50 = prev_williams_r > -50 and curr_williams_r <= -50
        crossed_above_50 = prev_williams_r < -50 and curr_williams_r >= -50
        
        # Exit conditions: opposite Williams %R crossover at midpoint
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
        
        # Entry conditions: Williams %R extreme with trend and volume filters
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold recovery) AND price above 1d EMA50 AND volume confirmation
            long_condition = crossed_above_80 and (curr_close > ema_50_level) and volume_confirm
            
            # Short: Williams %R crosses below -20 (overbought rejection) AND price below 1d EMA50 AND volume confirmation
            short_condition = crossed_below_20 and (curr_close < ema_50_level) and volume_confirm
            
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

name = "6h_WilliamsR_Reversal_1dEMA50Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0