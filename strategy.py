#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray with 1d regime filter
# Uses Williams Alligator (Jaw/Teeth/Lips) from 6h for trend direction and Elder Ray (Bull/Bear Power) from 1d for momentum confirmation.
# Only takes longs when: 6h price > Alligator Teeth (uptrend) AND 1d Bull Power > 0 AND 1d Bull Power > Bear Power (bullish momentum).
# Only takes shorts when: 6h price < Alligator Teeth (downtrend) AND 1d Bear Power < 0 AND 1d Bear Power < Bull Power (bearish momentum).
# Volume confirmation (>1.5x 20-period average) filters weak signals.
# Designed for ~15-30 trades/year on 6h timeframe to minimize fee drag while capturing high-probability moves.
# Works in both bull and bear markets via 1d Elder Ray regime filter - only trades when momentum aligns with trend.

name = "6h_WilliamsAlligator_ElderRay_1dEMA13_VolumeConfirm_v1"
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
    
    # Get 6h data for Williams Alligator (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator from 6h OHLC
    # Alligator Jaw (blue): 13-period SMMA smoothed 8 bars ahead
    # Alligator Teeth (red): 8-period SMMA smoothed 5 bars ahead  
    # Alligator Lips (green): 5-period SMMA smoothed 3 bars ahead
    # Using EMA as proxy for SMMA with proper alignment
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Jaw: 13-period EMA of median price, smoothed 8 bars
    median_6h = (high_6h + low_6h) / 2
    jaw_raw = pd.Series(median_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)  # smoothed 8 bars ahead
    jaw[:8] = jaw_raw[8] if len(jaw_raw) > 8 else jaw_raw[-1]  # fill initial values
    
    # Teeth: 8-period EMA of median price, smoothed 5 bars
    teeth_raw = pd.Series(median_6h).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)  # smoothed 5 bars ahead
    teeth[:5] = teeth_raw[5] if len(teeth_raw) > 5 else teeth_raw[-1]
    
    # Lips: 5-period EMA of median price, smoothed 3 bars
    lips_raw = pd.Series(median_6h).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)  # smoothed 3 bars ahead
    lips[:3] = lips_raw[3] if len(lips_raw) > 3 else lips_raw[-1]
    
    # Align Alligator lines to lower timeframe (with completed bar delay)
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    
    # Get 1d data for Elder Ray and regime filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    # Align Elder Ray components to lower timeframe (with completed bar delay)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate 20-period average volume for confirmation (on 6h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # Volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_13_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_jaw = jaw_aligned[i]
        curr_teeth = teeth_aligned[i]
        curr_lips = lips_aligned[i]
        curr_bull_power = bull_power_aligned[i]
        curr_bear_power = bear_power_aligned[i]
        curr_ema13_1d = ema_13_1d_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Determine Alligator trend: price > Teeth = uptrend, price < Teeth = downtrend
        # Alligator is aligned when Lips > Teeth > Jaw (uptrend) or Lips < Teeth < Jaw (downtrend)
        alligator_uptrend = curr_lips > curr_teeth and curr_teeth > curr_jaw
        alligator_downtrend = curr_lips < curr_teeth and curr_teeth < curr_jaw
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or Alligator turns against position (Lips cross below Jaw)
            if curr_close < entry_price - 2.5 * (curr_high - curr_low) or curr_lips < curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or Alligator turns against position (Lips cross above Jaw)
            if curr_close > entry_price + 2.5 * (curr_high - curr_low) or curr_lips > curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Long entry: Alligator uptrend AND Elder Ray bullish confirmation
            if vol_confirm and alligator_uptrend:
                if curr_bull_power > 0 and curr_bull_power > curr_bear_power:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
            # Short entry: Alligator downtrend AND Elder Ray bearish confirmation
            elif vol_confirm and alligator_downtrend:
                if curr_bear_power < 0 and abs(curr_bear_power) > abs(curr_bull_power):
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals