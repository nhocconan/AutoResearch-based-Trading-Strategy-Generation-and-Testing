#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator + 1d Regime Filter (ADX) + Volume Spike.
- Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength.
- 1d ADX > 25 filters for trending markets (avoids chop/range where Alligator whipsaws).
- Volume spike (>2x 20-period average) confirms breakout strength.
- Enter long when Lips > Teeth > Jaw (bullish alignment) + volume spike.
- Enter short when Lips < Teeth < Jaw (bearish alignment) + volume spike.
- Exit when Alligator reverses (Lips crosses Teeth) or volume drops.
- Position size 0.25 balances profit and drawdown control.
- Target trades: 80-160 total over 4 years (20-40/year) to balance opportunity and fee drag.
- Works in bull/bear markets via 1d ADX regime filter and volume confirmation logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ADX calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Williams Alligator on 4h (using SMMA - Smoothed Moving Average)
    def smma(data, period):
        """Smoothed Moving Average (Wilder's smoothing)"""
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(N-1) + CURRENT_DATA) / N
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator periods: Jaw=13, Teeth=8, Lips=5 (all shifted forward)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    # Smoothed median price (typical price)
    typical_price = (high + low + close) / 3
    
    jaw = smma(typical_price, jaw_period)
    teeth = smma(typical_price, teeth_period)
    lips = smma(typical_price, lips_period)
    
    # Shift the lines forward (Alligator characteristic)
    jaw = np.roll(jaw, -jaw_period)
    teeth = np.roll(teeth, -teeth_period)
    lips = np.roll(lips, -lips_period)
    
    # Volume confirmation: > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(jaw_period, teeth_period, lips_period, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: 1d ADX > 25 indicates trending market
        is_trending = adx_1d_aligned[i] > 25
        
        # Volume confirmation
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Alligator alignment conditions
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Only enter in trending markets with volume confirmation
            if is_trending and volume_confirm:
                # Long: bullish Alligator alignment
                if bullish_alignment:
                    signals[i] = 0.25
                    position = 1
                # Short: bearish Alligator alignment
                elif bearish_alignment:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Alligator reverses (Lips crosses below Teeth) OR market loses trend
            if lips[i] < teeth[i] or not is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator reverses (Lips crosses above Teeth) OR market loses trend
            if lips[i] > teeth[i] or not is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dADX_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0