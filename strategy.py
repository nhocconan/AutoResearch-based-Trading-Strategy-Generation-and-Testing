#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator with 1d EMA200 trend filter and 1d ATR volume spike.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA200 trend filter and ATR volume confirmation.
- Entry: Long when price > Alligator Jaw (TEMA13) AND Alligator Mouth is bullish (Lips > Teeth > Jaw) 
         AND price > 1d EMA200 AND ATR ratio > 2.0.
         Short when price < Alligator Jaw AND Alligator Mouth is bearish (Lips < Teeth < Jaw)
         AND price < 1d EMA200 AND ATR ratio > 2.0.
- Exit: Price crosses Alligator Jaw in opposite direction OR 1d EMA200 cross in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag.
- Williams Alligator uses Smoothed Moving Average (SMMA) with periods 13, 8, 5 and offsets 8, 5, 3.
- ATR ratio (current ATR/20-period ATR) > 2.0 confirms volatility expansion to avoid false signals.
- 1d EMA200 provides strong trend filter to avoid counter-trend trades in bear markets.
- Works in bull markets (follow Alligator uptrend) and bear markets (fade rallies below EMA200).
- Estimated trades: ~100 total over 4 years (~25/year) based on Alligator alignment frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Calculate Smoothed Moving Average (SMMA) - same as RMA/Wilder's smoothing."""
    return pd.Series(values).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period):
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First period
    return pd.Series(true_range).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

def alligator(close, jaw_period=13, teeth_period=8, lips_period=5, 
              jaw_offset=8, teeth_offset=5, lips_offset=3):
    """Calculate Williams Alligator: Jaw (TEMA13), Teeth (TEMA8), Lips (TEMA5)."""
    # Calculate SMMA for each period
    jaw_smma = smma(close, jaw_period)
    teeth_smma = smma(close, teeth_period)
    lips_smma = smma(close, lips_period)
    
    # Apply offsets (shift right by offset bars)
    jaw = np.roll(jaw_smma, jaw_offset)
    teeth = np.roll(teeth_smma, teeth_offset)
    lips = np.roll(lips_smma, lips_offset)
    
    # First 'offset' values are invalid due to roll
    jaw[:jaw_offset] = np.nan
    teeth[:teeth_offset] = np.nan
    lips[:lips_offset] = np.nan
    
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d trend filter: EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Calculate Williams Alligator on 6h close
    jaw, teeth, lips = alligator(close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 200)  # Need sufficient data for EMA200 and Alligator
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema200_1d_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Determine Alligator alignment
        alligator_bullish = lips[i] > teeth[i] > jaw[i]
        alligator_bearish = lips[i] < teeth[i] < jaw[i]
        
        # Exit conditions: price crosses Alligator Jaw OR 1d EMA200 cross in opposite direction
        if position != 0:
            # Exit long: price falls below Jaw OR price falls below 1d EMA200
            if position == 1:
                if curr_close < jaw[i] or curr_close < ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price rises above Jaw OR price rises above 1d EMA200
            elif position == -1:
                if curr_close > jaw[i] or curr_close > ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with trend and volatility filters
        if position == 0:
            # Long: bullish Alligator AND price > Jaw AND price > 1d EMA200 AND ATR ratio > 2.0
            if alligator_bullish and curr_close > jaw[i] and curr_close > ema200_1d_aligned[i] and atr_ratio_aligned[i] > 2.0:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator AND price < Jaw AND price < 1d EMA200 AND ATR ratio > 2.0
            elif alligator_bearish and curr_close < jaw[i] and curr_close < ema200_1d_aligned[i] and atr_ratio_aligned[i] > 2.0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA200_TrendFilter_1dATR_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0