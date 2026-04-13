#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# Uses Williams Alligator (Jaw, Teeth, Lips) to identify trend direction and strength.
# Daily timeframe filters trades to only align with higher timeframe trend.
# Volume confirmation ensures breakouts have conviction.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within profitable range.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on daily timeframe
    # Jaw (Blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (Red): 8-period SMMA, shifted 5 bars forward  
    # Lips (Green): 5-period SMMA, shifted 3 bars forward
    close_1d = df_1d['close'].values
    
    # Calculate SMMA (Smoothed Moving Average)
    def smma(data, period):
        if len(data) < period:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current Price) / Period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Shift the lines as per Williams Alligator specification
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted values that don't have enough data
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Alligator signals: 
    # When Lips > Teeth > Jaw -> Uptrend (Green > Red > Blue)
    # When Lips < Teeth < Jaw -> Downtrend (Green < Red < Blue)
    # When intertwined -> No trend (sleeping)
    uptrend = (lips_shifted > teeth_shifted) & (teeth_shifted > jaw_shifted)
    downtrend = (lips_shifted < teeth_shifted) & (teeth_shifted < jaw_shifted)
    
    # Calculate daily volume and its 20-period average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to 12-hour timeframe
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend.astype(float))
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend.astype(float))
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(uptrend_aligned[i]) or np.isnan(downtrend_aligned[i]) or
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 12h volume > 1.5x daily volume MA (adjusted for 12h)
        # 2 12h periods per day, so daily MA/2 = approximate 12h period MA
        volume_12h_approx_ma = volume_ma_20_1d_aligned[i] / 2
        volume_condition = volume[i] > (volume_12h_approx_ma * 1.5)
        
        # Entry conditions: Williams Alligator trend with volume confirmation
        # Long when uptrend aligned and volume confirmation
        # Short when downtrend aligned and volume confirmation
        if position == 0:
            if uptrend_aligned[i] > 0.5 and volume_condition:
                position = 1
                signals[i] = position_size
            elif downtrend_aligned[i] > 0.5 and volume_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when trend changes to downtrend or loses volume confirmation
            if downtrend_aligned[i] > 0.5 or not volume_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when trend changes to uptrend or loses volume confirmation
            if uptrend_aligned[i] > 0.5 or not volume_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Williams_Alligator_Volume_Filter_v2"
timeframe = "12h"
leverage = 1.0