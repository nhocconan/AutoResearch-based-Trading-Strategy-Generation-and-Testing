#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA(34) trend filter + volume confirmation
# Williams Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to identify
# trend absence (alligator sleeping) vs presence (alligator awake with mouth open).
# In ranging markets (alligator sleeping), we fade extreme deviations from the Jaw.
# In trending markets (alligator awake), we breakout in the direction of the trend.
# 1d EMA(34) ensures we trade with the daily trend to avoid counter-trend whipsaws.
# Volume confirmation filters low-participation moves.
# Designed for low trade frequency (12-37/year) to minimize fee drag.
# Works in both bull (trend following) and bear (mean reversion in ranges) markets.

name = "6h_WilliamsAlligator_1dEMA34_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA(34) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 6h timeframe (wait for completed 1d bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 6h data
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars
    # Lips: 5-period SMMA, shifted 3 bars
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.empty_like(data, dtype=float)
        result[:] = np.nan
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Close) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply shifts (Jaw: +8, Teeth: +5, Lips: +3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # First shifted values are invalid (set to NaN)
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Volume confirmation (2.0x 20-period average) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 35  # max(34 for 1d EMA, 13+8 for Jaw shift, 20 for volume MA +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Alligator conditions
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        
        # Alligator sleeping (ranging): all lines intertwined
        # Alligator awake (trending): lines separated, mouth open in direction of trend
        trending = (jaw_val > teeth_val > lips_val) or (jaw_val < teeth_val < lips_val)
        
        if position == 0:  # Flat - look for new entries
            if trending:
                # Trend following: breakout in direction of alligator alignment
                # Uptrend: jaw > teeth > lips
                # Downtrend: jaw < teeth < lips
                if jaw_val > teeth_val > lips_val:  # Uptrend
                    if close[i] > teeth_val and volume_spike[i]:  # Break above Teeth
                        signals[i] = 0.25
                        position = 1
                elif jaw_val < teeth_val < lips_val:  # Downtrend
                    if close[i] < teeth_val and volume_spike[i]:  # Break below Teeth
                        signals[i] = -0.25
                        position = -1
            else:
                # Ranging: mean reversion from extreme deviations
                # Fade when price deviates significantly from Jaw
                deviation = (close[i] - jaw_val) / jaw_val
                if deviation > 0.02 and volume_spike[i]:  # 2% above Jaw
                    signals[i] = -0.25  # Short fade
                    position = -1
                elif deviation < -0.02 and volume_spike[i]:  # 2% below Jaw
                    signals[i] = 0.25   # Long fade
                    position = 1
        elif position == 1:  # Long position
            # Exit: price closes below Lips (trend weakening) or opposite alligator alignment
            if close[i] < lips_shifted[i] or (jaw_shifted[i] < teeth_shifted[i] < lips_shifted[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above Lips (trend weakening) or opposite alligator alignment
            if close[i] > lips_shifted[i] or (jaw_shifted[i] > teeth_shifted[i] > lips_shifted[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals