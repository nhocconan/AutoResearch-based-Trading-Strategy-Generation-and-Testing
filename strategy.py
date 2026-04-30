#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA50 trend filter + volume confirmation.
# Uses Alligator's Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA) from 1d timeframe.
# In strong trends (price > 1d EMA50 and Alligator aligned bullish): long on pullback to Teeth with volume.
# In strong trends (price < 1d EMA50 and Alligator aligned bearish): short on pullback to Teeth with volume.
# In ranging markets (Alligator intertwined): fade at extremes with volume confirmation.
# Designed for low trade frequency (~12-30/year) to minimize fee drag on 6h timeframe.

name = "6h_1dWilliamsAlligator_1dEMA50_Trend_VolumePullback_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Williams Alligator and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator components (SMMA = Smoothed Moving Average)
    # SMMA formula: SMMA_t = (SMMA_{t-1} * (period-1) + close_t) / period
    # Initialize with SMA for first value
    close_1d = df_1d['close'].values
    
    def smma(arr, period):
        """Calculate Smoothed Moving Average"""
        sma = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return sma
        # First value is SMA
        sma[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA_t = (SMMA_{t-1} * (period-1) + close_t) / period
        for i in range(period, len(arr)):
            sma[i] = (sma[i-1] * (period-1) + arr[i]) / period
        return sma
    
    # Alligator: Jaw (13), Teeth (8), Lips (5) - all SMMA
    jaw = smma(close_1d, 13)   # Blue line
    teeth = smma(close_1d, 8)  # Red line
    lips = smma(close_1d, 5)   # Green line
    
    # Align Alligator components to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 100  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Regime filter: price above/below 1d EMA50 determines trend strength
        is_strong_uptrend = close[i] > ema_50_aligned[i]
        is_strong_downtrend = close[i] < ema_50_aligned[i]
        
        # Alligator alignment: bullish (Lips > Teeth > Jaw), bearish (Lips < Teeth < Jaw)
        is_alligator_bullish = (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i])
        is_alligator_bearish = (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i])
        is_alligator_ranging = not (is_alligator_bullish or is_alligator_bearish)
        
        curr_close = close[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            if is_strong_uptrend and is_alligator_bullish:
                # In strong uptrend: long on pullback to Teeth with volume confirmation
                if curr_close <= teeth_aligned[i] * 1.005 and curr_close >= teeth_aligned[i] * 0.995:
                    if curr_volume_spike:
                        signals[i] = 0.25
                        position = 1
                        entry_price = curr_close
            elif is_strong_downtrend and is_alligator_bearish:
                # In strong downtrend: short on pullback to Teeth with volume confirmation
                if curr_close <= teeth_aligned[i] * 1.005 and curr_close >= teeth_aligned[i] * 0.995:
                    if curr_volume_spike:
                        signals[i] = -0.25
                        position = -1
                        entry_price = curr_close
            elif is_alligator_ranging:
                # In ranging market: mean reversion at extremes with volume
                # Define extremes as 2 ATR from Alligator midline (average of all three)
                alligator_mid = (jaw_aligned[i] + teeth_aligned[i] + lips_aligned[i]) / 3
                # Approximate ATR using recent price range (simplified for 1d)
                # We'll use a fixed percentage for ranging extremes
                upper_extreme = alligator_mid * 1.02
                lower_extreme = alligator_mid * 0.98
                
                if curr_close <= lower_extreme and curr_volume_spike:
                    # Oversold: long
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                elif curr_close >= upper_extreme and curr_volume_spike:
                    # Overbought: short
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: price crosses above Lips (taking profit) or volume dries up
            if curr_close >= lips_aligned[i] or not curr_volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price crosses below Lips (taking profit) or volume dries up
            if curr_close <= lips_aligned[i] or not curr_volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals