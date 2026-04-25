#!/usr/bin/env python3
"""
1d_Alligator_Trend_WeeklyTrend_VolumeConfirm
Hypothesis: Williams Alligator on 1d timeframe with weekly trend filter and volume confirmation.
Long when Alligator is bullish (jaws < teeth < lips) with weekly uptrend and volume spike.
Short when Alligator is bearish (jaws > teeth > lips) with weekly downtrend and volume spike.
Alligator catches strong trends; weekly filter avoids counter-trend trades; volume confirms momentum.
Target: 15-25 trades/year (60-100 over 4 years) to minimize fee drag and work in both bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Williams Alligator: three smoothed moving averages
    # Jaw (blue): 13-period SMMA, shifted 8 bars ahead
    # Teeth (red): 8-period SMMA, shifted 5 bars ahead
    # Lips (green): 5-period SMMA, shifted 3 bars ahead
    # SMMA = smoothed moving average (similar to EMA but with different smoothing)
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate Alligator lines on daily data
    jaw = smma(close_1d, 13)  # 13-period SMMA
    teeth = smma(close_1d, 8)  # 8-period SMMA
    lips = smma(close_1d, 5)   # 5-period SMMA
    
    # Shift the lines: Jaw shifted 8, Teeth shifted 5, Lips shifted 3
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Set NaN for shifted positions that don't have data
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Align Alligator lines to 1d timeframe (they're already on 1d, just need alignment for look-ahead safety)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    # Weekly EMA34 for trend filter (using close prices)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Alligator (max shift 8), weekly EMA34, and volume MA
    start_idx = max(8 + 13, 34, 20)  # jaw period + shift, weekly EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Alligator conditions
        bullish_alligator = (jaw_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < lips_aligned[i])
        bearish_alligator = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_34_1w_aligned[i]
        weekly_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: bullish Alligator + weekly uptrend + volume spike
            long_setup = bullish_alligator and weekly_uptrend and volume_spike[i]
            # Short: bearish Alligator + weekly downtrend + volume spike
            short_setup = bearish_alligator and weekly_downtrend and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: Alligator turns bearish OR weekly trend turns down
            if (not bullish_alligator) or (not weekly_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Alligator turns bullish OR weekly trend turns up
            if (not bearish_alligator) or (not weekly_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Alligator_Trend_WeeklyTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0