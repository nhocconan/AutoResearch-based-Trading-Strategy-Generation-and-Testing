#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams Alligator (Jaw/Teeth/Lips) with volume confirmation and chop regime filter.
# Enter long when Alligator lines are bullish (Lips > Teeth > Jaw) with volume spike and chop < 61.8 (trending).
# Enter short when Alligator lines are bearish (Lips < Teeth < Jaw) with volume spike and chop < 61.8.
# Uses discrete position sizing (0.25) to balance return and drawdown. Target: 20-50 trades/year.
# Alligator provides trend direction from higher timeframe, volume confirms strength, chop filter avoids ranging markets.
# Works in bull (trend continuation) and bear (trend reversal) markets by following the Alligator's alignment.

name = "4h_Alligator_Trend_Volume_ChopFilter_v1"
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
    
    # Get 1d data for Williams Alligator (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Williams Alligator: SMAs of median price
    # Jaw: 13-period SMA, shifted 8 bars ahead
    # Teeth: 8-period SMA, shifted 5 bars ahead
    # Lips: 5-period SMA, shifted 3 bars ahead
    median_price = (df_1d['high'] + df_1d['low']) / 2.0
    close_1d = df_1d['close'].values
    
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align 1d indicators to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_raw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_raw)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_raw)
    
    # Calculate 4h chop regime: EHLERS CHOPPINESS INDEX (14)
    def choppiness_index(high, low, close, length=14):
        atr_sum = np.zeros_like(close)
        true_range = np.zeros_like(close)
        for i in range(1, len(close)):
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            true_range[i] = tr
            if i >= length:
                atr_sum[i] = atr_sum[i-1] + tr - true_range[i-length+1]
            else:
                atr_sum[i] = atr_sum[i-1] + tr
        atr = atr_sum / length
        max_high = np.zeros_like(close)
        min_low = np.zeros_like(close)
        for i in range(len(close)):
            if i < length:
                max_high[i] = np.max(high[:i+1])
                min_low[i] = np.min(low[:i+1])
            else:
                max_high[i] = np.max(high[i-length+1:i+1])
                min_low[i] = np.min(low[i-length+1:i+1])
        chop = np.zeros_like(close)
        for i in range(length-1, len(close)):
            if max_high[i] != min_low[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(length)
            else:
                chop[i] = 50.0
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    chop_trending = chop < 61.8  # Trending regime when chop < 61.8
    
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
            np.isnan(volume_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Alligator trend conditions with volume confirmation and chop filter
        # Bullish: Lips > Teeth > Jaw
        bullish = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        # Bearish: Lips < Teeth < Jaw
        bearish = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        long_entry = bullish and volume_spike[i] and chop_trending[i]
        short_entry = bearish and volume_spike[i] and chop_trending[i]
        
        # Exit conditions: opposite Alligator alignment
        long_exit = bearish  # Exit long when Alligator turns bearish
        short_exit = bullish  # Exit short when Alligator turns bullish
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
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