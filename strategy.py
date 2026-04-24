#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 12h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h for EMA50 trend direction.
- Williams %R: Measures overbought/oversold on 6h chart (period=14).
- Entry: Long when Williams %R crosses above -80 from below AND price > 12h EMA50 AND volume > 2.0 * 20-period average volume.
         Short when Williams %R crosses below -20 from above AND price < 12h EMA50 AND volume > 2.0 * 20-period average volume.
- Exit: Opposite Williams %R cross (long exit when crosses below -50, short exit when crosses above -50).
- Signal size: 0.25 discrete to minimize fee drag.
- Williams %R is effective in ranging markets; trend filter avoids counter-trend trades.
- Volume spike confirms institutional participation, reducing false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def williams_r(high, low, close, period):
    """Calculate Williams %R."""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    wr = np.where((highest_high - lowest_low) == 0, -50, wr)
    return wr

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Williams %R (period=14)
    wr = williams_r(high, low, close, 14)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema50_12h = ema(df_12h['close'].values, 50)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 6h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for volume MA, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(wr[i]) or np.isnan(wr[i-1]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_wr = wr[i]
        prev_wr = wr[i-1]
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Williams %R cross conditions
        wr_cross_above_80 = prev_wr <= -80 and curr_wr > -80
        wr_cross_below_20 = prev_wr >= -20 and curr_wr < -20
        wr_cross_above_50 = prev_wr <= -50 and curr_wr > -50
        wr_cross_below_50 = prev_wr >= -50 and curr_wr < -50
        
        # Exit conditions: opposite Williams %R cross at -50 level
        if position != 0:
            # Exit long: Williams %R crosses below -50
            if position == 1:
                if wr_cross_below_50:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R crosses above -50
            elif position == -1:
                if wr_cross_above_50:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R with trend filter and volume confirmation
        if position == 0:
            # Long: Williams %R crosses above -80 from below AND price > 12h EMA50
            long_condition = (wr_cross_above_80 and 
                            curr_close > ema50_12h_aligned[i] and
                            volume_confirm)
            
            # Short: Williams %R crosses below -20 from above AND price < 12h EMA50
            short_condition = (wr_cross_below_20 and 
                             curr_close < ema50_12h_aligned[i] and
                             volume_confirm)
            
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

name = "6h_WilliamsR_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0