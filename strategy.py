#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA50 trend filter + volume confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) from 6h timeframe to identify trend
# Only trade in direction of 1d EMA50 trend (bull/bear filter)
# Volume spike (1.8x 20-period average) confirms institutional participation
# Alligator provides dynamic support/resistance in trending markets
# Works in bull markets via buying on pullbacks to Teeth in uptrends
# Works in bear markets via selling on rallies to Teeth in downtrends
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Williams_Alligator_1dEMA50_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 6h data
    # Jaw: Blue line (13-period SMMA, shifted 8 bars ahead)
    # Teeth: Red line (8-period SMMA, shifted 5 bars ahead)
    # Lips: Green line (5-period SMMA, shifted 3 bars ahead)
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + PRICE) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift the lines as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # First few values become invalid after rolling, set to NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 8, 5) + 8  # warmup for Alligator calculation
    
    for i in range(start_idx, n):
        # Skip if any Alligator line is NaN
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: volume > 1.8x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (1.8 * vol_ma_20)
        
        curr_close = close[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        
        # Determine Alligator alignment (trend direction)
        # Mouth open (trending): Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish)
        # Mouth closed (ranging): lines intertwined
        bullish_alignment = curr_lips > curr_teeth > curr_jaw
        bearish_alignment = curr_lips < curr_teeth < curr_jaw
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and clear Alligator alignment
            if volume_spike:
                # Bullish entry: price above Teeth AND bullish alignment AND above 1d EMA50
                if curr_close > curr_teeth and bullish_alignment and curr_close > curr_ema_50_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price below Teeth AND bearish alignment AND below 1d EMA50
                elif curr_close < curr_teeth and bearish_alignment and curr_close < curr_ema_50_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below Lips or Alligator alignment breaks
            if curr_close < curr_lips or not bullish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above Lips or Alligator alignment breaks
            if curr_close > curr_lips or not bearish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals