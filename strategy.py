#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d volume confirmation + 1d trend filter
# - Williams Alligator (Jaw=13, Teeth=8, Lips=5) defines trend on 4h: 
#   * Lips > Teeth > Jaw = bullish alignment (long bias)
#   * Lips < Teeth < Jaw = bearish alignment (short bias)
# - 1d volume > 1.3x 20-period average for conviction (avoid low-volume false signals)
# - 1d close > 1d EMA50 for long bias, < EMA50 for short bias (trend filter)
# - Entry: Alligator alignment + 1d volume + 1d trend in same direction
# - Exit: Opposite Alligator alignment or volume drops below average
# - Position size: 0.25 (25%) to balance return and drawdown
# - Designed for both bull and bear markets by following higher timeframe trend
# - Target: ~25 trades/year to minimize fee drag (<2.5% annual)

name = "4h_WilliamsAlligator_1dVolume_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Williams Alligator
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Williams Alligator components (Smoothed SMAs)
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw_raw = pd.Series(close_4h).rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.shift(8)
    
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth_raw = pd.Series(close_4h).rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.shift(5)
    
    # Lips: 5-period SMMA, shifted 3 bars
    lips_raw = pd.Series(close_4h).rolling(window=5, min_periods=5).mean()
    lips = lips_raw.shift(3)
    
    # Align Alligator lines to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips.values)
    
    # Get 1d data for volume and trend filters
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d volume average (20-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator alignment signals
        lips_val = lips_aligned[i]
        teeth_val = teeth_aligned[i]
        jaw_val = jaw_aligned[i]
        
        bullish_alignment = lips_val > teeth_val and teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val and teeth_val < jaw_val
        
        # 1d volume filter: current volume > 1.3x average
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.3 * vol_ma_1d_aligned[i]
        
        # 1d trend filter
        above_ema = close[i] > ema_50_1d_aligned[i]
        below_ema = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: bullish alignment + volume + above EMA50
            if bullish_alignment and volume_filter and above_ema:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish alignment + volume + below EMA50
            elif bearish_alignment and volume_filter and below_ema:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on bearish alignment or volume drops
            if bearish_alignment or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on bullish alignment or volume drops
            if bullish_alignment or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals