#!/usr/bin/env python3
"""
12h Williams Alligator + 1d EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator (JAW/TEETH/LIPS) identifies trend direction and momentum on 12h.
1d EMA50 provides higher-timeframe trend filter to avoid counter-trend trades.
Volume spike confirms breakout strength. Works in bull/bear by following 1d trend.
Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator on 12h (primary timeframe)
    # JAW: 13-period SMMA, shifted 8 bars
    # TEETH: 8-period SMMA, shifted 5 bars  
    # LIPS: 5-period SMMA, shifted 3 bars
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full_like(data, np.nan)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CLOSE) / N
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift to avoid look-ahead (Alligator lines are shifted forward)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Fill shifted values with NaN
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13)  # Need 50 for daily EMA, 20 for volume MA, 13 for Alligator
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        ema_50_level = ema_50_1d_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average volume
        volume_spike = curr_volume > 2.0 * vol_ma_20[i]
        
        # Alligator trend conditions
        # Bullish: Lips > Teeth > Jaw (green, aligned up)
        bullish_aligned = lips_val > teeth_val and teeth_val > jaw_val
        # Bearish: Lips < Teeth < Jaw (red, aligned down)
        bearish_aligned = lips_val < teeth_val and teeth_val < jaw_val
        
        # Exit conditions: opposite Alligator alignment
        if position != 0:
            # Exit long: bearish alignment
            if position == 1:
                if bearish_aligned:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: bullish alignment
            elif position == -1:
                if bullish_aligned:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with trend and volume filters
        if position == 0:
            # Long: bullish alignment AND above daily EMA50 AND volume spike
            long_condition = bullish_aligned and (curr_close > ema_50_level) and volume_spike
            
            # Short: bearish alignment AND below daily EMA50 AND volume spike
            short_condition = bearish_aligned and (curr_close < ema_50_level) and volume_spike
            
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

name = "12h_Williams_Alligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0