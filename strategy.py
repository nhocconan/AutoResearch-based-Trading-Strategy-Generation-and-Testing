#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA(34) trend filter and volume confirmation
# Williams Alligator consists of three smoothed moving averages (Jaw, Teeth, Lips) that
# indicate trend direction and strength. When the lines are intertwined (sleeping),
# the market is ranging; when they diverge (awakening), a trend is forming.
# We trade in the direction of the 1d EMA(34) trend only when the Alligator is
# awakening (teeth above/below lips) and confirmed by volume spike.
# Designed for low trade frequency (12-37/year) to minimize fee drag. Works in both
# bull and bear markets by filtering counter-trend signals and using volume
# confirmation to avoid false breakouts.

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
    
    # Williams Alligator on 6h timeframe
    # Jaw: 13-period SMMA, shifted 8 bars ahead
    # Teeth: 8-period SMMA, shifted 5 bars ahead
    # Lips: 5-period SMMA, shifted 3 bars ahead
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 1.0 / period
        result = np.empty_like(arr)
        result[:] = np.nan
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (arr[i] * alpha) + (result[i-1] * (1 - alpha))
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Fill shifted values with NaN
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Volume confirmation (2.0x 20-period average) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 35  # max(13+8, 8+5, 5+3, 34 for 1d EMA, 20 for volume MA +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Teeth above Lips (bullish alignment) + price above 1d EMA(34) + volume spike
            if (teeth_shifted[i] > lips_shifted[i] and close[i] > ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Teeth below Lips (bearish alignment) + price below 1d EMA(34) + volume spike
            elif (teeth_shifted[i] < lips_shifted[i] and close[i] < ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Teeth below Lips (trend reversal) or price below 1d EMA(34) (trend reversal)
            if teeth_shifted[i] < lips_shifted[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Teeth above Lips (trend reversal) or price above 1d EMA(34) (trend reversal)
            if teeth_shifted[i] > lips_shifted[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals