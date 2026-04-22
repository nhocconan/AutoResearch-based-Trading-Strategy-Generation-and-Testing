#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 12h trend filter and volume confirmation.
# Uses Williams Alligator (Jaw/Teeth/Lips) to identify trend direction and strength.
# Only trades in strong trends (JAW > TEETH > LIPS for long, JAW < TEETH < LIPS for short).
# Adds 12h EMA50 trend filter and volume spike confirmation to reduce false signals.
# Designed to work in both bull and bear markets by following major trends.
# Targets 20-50 trades/year with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams Alligator parameters
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    # Calculate median price
    median_price = (prices['high'].values + prices['low'].values) / 2
    
    # Calculate Alligator lines (SMMA of median price)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) >= period:
            # First value is SMA
            result[period-1] = np.mean(arr[:period])
            # Subsequent values are SMMA
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, jaw_period)
    teeth = smma(median_price, teeth_period)
    lips = smma(median_price, lips_period)
    
    # Apply shifts
    jaw = np.roll(jaw, jaw_shift)
    teeth = np.roll(teeth, teeth_shift)
    lips = np.roll(lips, lips_shift)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_50_12h = ema_50_12h_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        # Alligator alignment checks
        bullish_alignment = jaw_val > teeth_val > lips_val
        bearish_alignment = jaw_val < teeth_val < lips_val
        
        if position == 0:
            # Look for strong trend with volume confirmation
            if bullish_alignment and price > ema_50_12h and vol_spike:
                signals[i] = 0.25
                position = 1
            elif bearish_alignment and price < ema_50_12h and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Alligator lines cross or price crosses 12h EMA50
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on bearish alignment or price below 12h EMA50
                if not bullish_alignment or price < ema_50_12h:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on bullish alignment or price above 12h EMA50
                if not bearish_alignment or price > ema_50_12h:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsAlligator_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0