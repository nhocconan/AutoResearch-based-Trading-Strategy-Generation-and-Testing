#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d ADX filter and volume confirmation.
# Uses Williams Alligator (Jaw/Teeth/Lips) to identify trend direction and strength.
# 1d ADX > 25 filters for trending markets to avoid whipsaws in ranging conditions.
# Volume confirmation (>1.5x 20-period average) reduces false signals.
# Entry when Lips cross above/below Teeth in direction of trend with volume confirmation.
# Exit when Lips cross back below/above Teeth or trend weakens (ADX < 20).
# Designed to work in both bull and bear markets by using 1d trend filter to avoid counter-trend trades.
# Target: 20-25 trades/year per symbol (80-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ADX on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_period = 14
    atr = pd.Series(tr).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Calculate Williams Alligator on price data (using close prices)
    # Jaw: Blue line - 13-period SMMA smoothed 8 periods ahead
    # Teeth: Red line - 8-period SMMA smoothed 5 periods ahead  
    # Lips: Green line - 5-period SMMA smoothed 3 periods ahead
    
    def smma(data, period):
        """Smoothed Moving Average"""
        sma = pd.Series(data).rolling(window=period, min_periods=period).mean().values
        # First value is SMA, then recursive smoothing
        result = np.full_like(data, np.nan)
        result[period-1] = sma[period-1]
        for i in range(period, len(data)):
            if not np.isnan(sma[i]):
                result[i] = (result[i-1] * (period-1) + sma[i]) / period
        return result
    
    jaw = smma(close, 13)  # 13-period SMMA
    teeth = smma(close, 8)  # 8-period SMMA
    lips = smma(close, 5)   # 5-period SMMA
    
    # Apply the forward shifts as per Williams Alligator definition
    jaw = np.roll(jaw, 8)   # Shifted 8 periods ahead
    teeth = np.roll(teeth, 5) # Shifted 5 periods ahead
    lips = np.roll(lips, 3)   # Shifted 3 periods ahead
    
    # Align indicators
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    jaw_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), jaw)
    teeth_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), teeth)
    lips_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), lips)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(13 + 8, 20)  # Need Alligator and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Look for Alligator crossovers
            # Only trade in trending markets
            
            # Long: Lips cross above Teeth AND trending market
            if (lips_aligned[i] > teeth_aligned[i] and 
                lips_aligned[i-1] <= teeth_aligned[i-1] and  # Crossover confirmation
                trending and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: Lips cross below Teeth AND trending market
            elif (lips_aligned[i] < teeth_aligned[i] and 
                  lips_aligned[i-1] >= teeth_aligned[i-1] and  # Crossover confirmation
                  trending and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Lips cross back below Teeth or trend weakens
            if (lips_aligned[i] < teeth_aligned[i] or 
                adx_aligned[i] < 20):  # Trend weakening
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Lips cross back above Teeth or trend weakens
            if (lips_aligned[i] > teeth_aligned[i] or 
                adx_aligned[i] < 20):  # Trend weakening
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dADX_WilliamsAlligator_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0