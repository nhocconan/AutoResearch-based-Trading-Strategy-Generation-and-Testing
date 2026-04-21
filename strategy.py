#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d ADX Trend Filter + Volume Spike
# Williams Alligator: Jaw (SMA13, 8 shift), Teeth (SMA8, 5 shift), Lips (SMA5, 3 shift)
# Long when Lips > Teeth > Jaw in uptrend (price > 1d ADX > 25 and DI+ > DI-)
# Short when Lips < Teeth < Jaw in downtrend (price > 1d ADX > 25 and DI- > DI+)
# Volume spike (>1.8x 20-period average) confirms conviction
# Works in bull/bear: 1d ADX ensures we only trade when trending, avoids chop
# Target: 20-35 trades/year by requiring ADX trend + Alligator alignment + volume

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI and DX
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilders_smoothing(dx, 14)
    
    # Handle division by zero and NaN
    adx = np.where((di_plus + di_minus) == 0, 0, adx)
    adx = np.nan_to_num(adx, nan=0.0)
    
    adx_25 = adx > 25
    di_plus_gt_di_minus = di_plus > di_minus
    di_minus_gt_di_plus = di_minus > di_plus
    
    adx_25_aligned = align_htf_to_ltf(prices, df_1d, adx_25)
    di_plus_gt_di_minus_aligned = align_htf_to_ltf(prices, df_1d, di_plus_gt_di_minus)
    di_minus_gt_di_plus_aligned = align_htf_to_ltf(prices, df_1d, di_minus_gt_di_plus)
    
    # Calculate Williams Alligator (on 4h data)
    close = prices['close'].values
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Alligator alignment conditions
    lips_above_teeth = lips > teeth
    teeth_above_jaw = teeth > jaw
    lips_below_teeth = lips < teeth
    teeth_below_jaw = teeth < jaw
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if data not ready
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirm = volume > 1.8 * vol_ma[i]
        
        # Trend filter from 1d ADX
        uptrend = adx_25_aligned[i] and di_plus_gt_di_minus_aligned[i]
        downtrend = adx_25_aligned[i] and di_minus_gt_di_plus_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Long: Lips > Teeth > Jaw in uptrend
                if lips_above_teeth[i] and teeth_above_jaw[i] and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Lips < Teeth < Jaw in downtrend
                elif lips_below_teeth[i] and teeth_below_jaw[i] and downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Alligator alignment breaks or trend fails
                if not (lips_above_teeth[i] and teeth_above_jaw[i]) or not uptrend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Alligator alignment breaks or trend fails
                if not (lips_below_teeth[i] and teeth_below_jaw[i]) or not downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dADX_Trend_Volume"
timeframe = "4h"
leverage = 1.0