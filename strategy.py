#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d ADX trend filter and volume confirmation.
# Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
# Alligator lines intertwined = no trend (sleeping), diverged = trend (awake)
# Strategy: Go long when Lips > Teeth > Jaw (bullish alignment) + ADX > 25 (trending) + volume spike
# Go short when Lips < Teeth < Jaw (bearish alignment) + ADX > 25 (trending) + volume spike
# Exit when Alligator lines re-intertwine (Lips crosses Teeth) or ADX < 20 (trend weakens)
# Designed for ~20-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator calculation (using smoothed moving average - SMMA)
    def smma(source, period):
        """Smoothed Moving Average"""
        sma = pd.Series(source).rolling(window=period, min_periods=period).mean().values
        smma_vals = np.full_like(source, np.nan, dtype=float)
        if len(source) >= period:
            smma_vals[period-1] = sma[period-1]
            for i in range(period, len(source)):
                if not np.isnan(smma_vals[i-1]):
                    smma_vals[i] = (smma_vals[i-1] * (period-1) + source[i]) / period
        return smma_vals
    
    # Alligator lines
    jaw = smma(close, 13)  # Jaw (Blue): 13-period SMMA shifted 8 bars
    teeth = smma(close, 8)  # Teeth (Red): 8-period SMMA shifted 5 bars
    lips = smma(close, 5)   # Lips (Green): 5-period SMMA shifted 3 bars
    
    # Apply shifts (Jaw: +8, Teeth: +5, Lips: +3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Invalidate the shifted values that look ahead
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX calculation on 1d data
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed TR, +DM, -DM
        atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=period).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=period).mean().values / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = np.where((plus_di + minus_di) == 0, 0, dx)
        adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(lips_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment conditions
        lips_above_teeth = lips_shifted[i] > teeth_shifted[i]
        teeth_above_jaw = teeth_shifted[i] > jaw_shifted[i]
        bullish_alignment = lips_above_teeth and teeth_above_jaw
        
        lips_below_teeth = lips_shifted[i] < teeth_shifted[i]
        teeth_below_jaw = teeth_shifted[i] < jaw_shifted[i]
        bearish_alignment = lips_below_teeth and teeth_below_jaw
        
        # ADX trend filter
        strong_trend = adx_1d_aligned[i] > 25
        weak_trend = adx_1d_aligned[i] < 20
        
        # Entry conditions
        if bullish_alignment and strong_trend and volume_filter[i]:
            signals[i] = 0.25
            position = 1
        elif bearish_alignment and strong_trend and volume_filter[i]:
            signals[i] = -0.25
            position = -1
        # Exit conditions: Alligator lines re-intertwine or trend weakens
        elif position == 1 and (not bullish_alignment or weak_trend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not bearish_alignment or weak_trend):
            signals[i] = 0.0
            position = 0
        # Hold position
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_WilliamsAlligator_1dADX25_VolumeFilter"
timeframe = "4h"
leverage = 1.0