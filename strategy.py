#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA50 trend direction.
- Williams %R(14): Oversold < -80 for long, Overbought > -20 for short.
- Trend Filter: 1d EMA50 must align with reversal direction (long: close > EMA50, short: close < EMA50).
- Volume Filter: Current 6h volume > 1.8 * 20-period average 6h volume to confirm momentum.
- Entry: Long when Williams %R crosses above -80 AND close > 1d EMA50 AND volume spike.
         Short when Williams %R crosses below -20 AND close < 1d EMA50 AND volume spike.
- Exit: Opposite Williams %R level (long exits when %R crosses below -20, short exits when %R crosses above -80).
- Signal size: 0.25 discrete to minimize fee drag.
- Designed to capture mean-reversion reversals aligned with daily trend, working in both bull (buy dips) and bear (sell rallies).
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) on 6h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 6h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14, 20)  # Need 50 for EMA, 14 for Williams %R, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_williams_r = williams_r[i]
        prev_williams_r = williams_r[i-1] if i > 0 else -50
        ema_50_level = ema_50_1d_aligned[i]
        
        # Volume spike: current volume > 1.8 * 20-period average volume
        volume_spike = curr_volume > 1.8 * vol_ma_20[i]
        
        # Williams %R crossover conditions
        crossed_above_oversold = (prev_williams_r <= -80) and (curr_williams_r > -80)
        crossed_below_overbought = (prev_williams_r >= -20) and (curr_williams_r < -20)
        
        # Trend alignment conditions
        above_ema = curr_close > ema_50_level
        below_ema = curr_close < ema_50_level
        
        # Exit conditions: opposite Williams %R level
        if position != 0:
            # Exit long: Williams %R crosses below overbought (-20)
            if position == 1:
                if curr_williams_r < -20 and prev_williams_r >= -20:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R crosses above oversold (-80)
            elif position == -1:
                if curr_williams_r > -80 and prev_williams_r <= -80:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R reversal with trend and volume filters
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold) AND above EMA50 AND volume spike
            long_condition = crossed_above_oversold and above_ema and volume_spike
            
            # Short: Williams %R crosses below -20 (overbought) AND below EMA50 AND volume spike
            short_condition = crossed_below_overbought and below_ema and volume_spike
            
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

name = "6h_WilliamsR_Reversal_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0