#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams Alligator (Jaw/Teeth/Lips) as trend filter,
# combined with 4h Donchian(20) breakout and volume confirmation.
# Alligator: Jaw (13-period SMMA, 8-bar shift), Teeth (8-period SMMA, 5-bar shift), Lips (5-period SMMA, 3-bar shift).
# Trend: Lips > Teeth > Jaw = bullish; Lips < Teeth < Jaw = bearish.
# Breakout: long when price > Donchian(20) high + volume spike; short when price < Donchian(20) low + volume spike.
# Volume spike: >2.0x 20-bar average volume.
# Position size 0.25 balances return and drawdown. Discrete levels minimize fee churn.
# Target: 75-200 total trades over 4 years = 19-50/year for 4h (within proven winning range).

name = "4h_WilliamsAlligator_Donchian20_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Smoothed Moving Average (SMMA) calculation
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=np.float64)
        result = np.full_like(arr, np.nan, dtype=np.float64)
        # First value is simple moving average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Williams Alligator lines
    jaw = smma(close_1d, 13)  # 13-period SMMA
    teeth = smma(close_1d, 8)  # 8-period SMMA
    lips = smma(close_1d, 5)   # 5-period SMMA
    
    # Apply Alligator shifts (Jaw: 8 bars, Teeth: 5 bars, Lips: 3 bars)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted positions that roll in invalid data
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Align Alligator lines to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    # Calculate 4h Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator trend filter
        bullish = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Donchian breakout conditions with volume confirmation
        long_breakout = close[i] > donchian_high[i] and volume_spike[i]
        short_breakout = close[i] < donchian_low[i] and volume_spike[i]
        
        # Exit conditions: opposite Donchian level or trend reversal
        long_exit = close[i] < donchian_low[i] or not bullish
        short_exit = close[i] > donchian_high[i] or not bearish
        
        # Handle entries and exits
        if long_breakout and bullish and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and bearish and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals