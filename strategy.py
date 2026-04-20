#!/usr/bin/env python3
"""
12h_1d_WilliamsAlligator_Trend_v1
Concept: 12h Williams Alligator trend-following with daily volume confirmation and ADX filter.
- Long: Price > Alligator Jaw AND Alligator Mouth > 0 AND ADX > 25 AND daily volume > 1.2x 20-period avg
- Short: Price < Alligator Jaw AND Alligator Mouth < 0 AND ADX > 25 AND daily volume > 1.2x 20-period avg
- Exit: Price crosses back through Alligator Teeth
- Williams Alligator: Jaw (13-period SMA shifted 8), Teeth (8-period SMA shifted 5), Lips (5-period SMA shifted 3)
- Position sizing: 0.25
- Target: 50-150 total trades over 4 years (12-37/year)
- Works in bull/bear: ADX ensures trending markets, volume confirms conviction, Alligator avoids whipsaws
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_WilliamsAlligator_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily: Volume Spike Filter ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 12h: Williams Alligator ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Alligator components (Smoothed Moving Average with shifts)
    # Jaw: 13-period SMMA shifted 8 bars forward
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.shift(8)
    
    # Teeth: 8-period SMMA shifted 5 bars forward
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.shift(5)
    
    # Lips: 5-period SMMA shifted 3 bars forward
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips_raw.shift(3)
    
    # Alligator Mouth: difference between Jaw and Teeth (positive = bullish alignment)
    alligator_mouth = jaw - teeth
    
    # === 12h: ADX (20-period) for trend strength ===
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed averages
    def wilders_smooth(data, period):
        """Wilder's smoothing (exponential with alpha=1/period)"""
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smooth(tr, 20)
    plus_di = 100 * wilders_smooth(plus_dm, 20) / atr
    minus_di = 100 * wilders_smooth(minus_dm, 20) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, 20)
    
    # Align all indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips.values)
    alligator_mouth_aligned = align_htf_to_ltf(prices, df_1d, alligator_mouth)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        mouth_val = alligator_mouth_aligned[i]
        adx_val = adx_aligned[i]
        close_val = close[i]
        
        # Skip if any value is NaN
        if (np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val) or 
            np.isnan(mouth_val) or np.isnan(adx_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current daily volume > 1.2x 20-period average
        vol_1d_vals = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_vals)
        current_vol = vol_1d_aligned[i]
        vol_condition = current_vol > 1.2 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price above jaw, bullish alignment, strong trend, volume confirmation
            if close_val > jaw_val and mouth_val > 0 and adx_val > 25 and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price below jaw, bearish alignment, strong trend, volume confirmation
            elif close_val < jaw_val and mouth_val < 0 and adx_val > 25 and vol_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below teeth (or lips for earlier exit)
            if close_val < teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above teeth (or lips for earlier exit)
            if close_val > teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals