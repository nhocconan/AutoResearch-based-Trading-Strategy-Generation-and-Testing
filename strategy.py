#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# - Williams Alligator (jaw=13, teeth=8, lips=5) from 1d: aligned = bullish, reversed = bearish
# - 1d EMA(50) > EMA(200) for bullish trend, < for bearish trend (avoid counter-trend)
# - Volume confirmation: current 12h volume > 1.8x 30-period average to confirm participation
# - Designed for 12h timeframe: targets 12-37 trades/year (50-150 total over 4 years)
# - Works in bull/bear markets: 1d EMA filter ensures we trade with higher timeframe trend
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "12h_1d_alligator_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Williams Alligator from 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    median_1d = (high_1d + low_1d) / 2.0
    
    # Alligator lines: jaw(13), teeth(8), lips(5) - all smoothed with SMMA
    def smma(arr, period):
        """Smoothed Moving Average (similar to Wilder's smoothing)"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_smma * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_1d, 13)  # Blue line
    teeth = smma(median_1d, 8)  # Red line
    lips = smma(median_1d, 5)   # Green line
    
    # Alligator alignment: bullish when lips > teeth > jaw, bearish when lips < teeth < jaw
    alligator_bullish = (lips > teeth) & (teeth > jaw)
    alligator_bearish = (lips < teeth) & (teeth < jaw)
    
    # Align Alligator signals to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    alligator_bullish_aligned = align_htf_to_ltf(prices, df_1d, alligator_bullish.astype(float))
    alligator_bearish_aligned = align_htf_to_ltf(prices, df_1d, alligator_bearish.astype(float))
    
    # Pre-compute 1d EMA trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_bullish = ema_50 > ema_200
    ema_bearish = ema_50 < ema_200
    
    ema_bullish_aligned = align_htf_to_ltf(prices, df_1d, ema_bullish.astype(float))
    ema_bearish_aligned = align_htf_to_ltf(prices, df_1d, ema_bearish.astype(float))
    
    # Pre-compute 12h volume confirmation
    volume_12h = prices['volume'].values
    avg_volume_30 = pd.Series(volume_12h).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume_12h > (1.8 * avg_volume_30)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(alligator_bullish_aligned[i]) or np.isnan(alligator_bearish_aligned[i]) or
            np.isnan(ema_bullish_aligned[i]) or np.isnan(ema_bearish_aligned[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Alligator turns bearish or EMA trend turns bearish
            if alligator_bearish_aligned[i] > 0.5 or ema_bearish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish or EMA trend turns bullish
            if alligator_bullish_aligned[i] > 0.5 or ema_bullish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Alligator alignment with trend and volume filters
            if vol_spike[i]:
                # Bullish setup: Alligator bullish + EMA bullish
                if alligator_bullish_aligned[i] > 0.5 and ema_bullish_aligned[i] > 0.5:
                    position = 1
                    signals[i] = 0.25
                # Bearish setup: Alligator bearish + EMA bearish
                elif alligator_bearish_aligned[i] > 0.5 and ema_bearish_aligned[i] > 0.5:
                    position = -1
                    signals[i] = -0.25
    
    return signals