#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume spike and price above/below teeth.
# Long when green line > red line > blue line + volume spike + price > teeth.
# Short when green line < red line < blue line + volume spike + price < teeth.
# Williams Alligator uses smoothed SMAs: Jaw(13,8), Teeth(8,5), Lips(5,3).
# Trend-following indicator that works in both bull and bear markets by capturing strong trends.
# Target: 15-25 trades/year to minimize fee drag while capturing major moves.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: three smoothed SMAs
    # Jaw (Blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (Red): 8-period SMMA, shifted 5 bars forward  
    # Lips (Green): 5-period SMMA, shifted 3 bars forward
    # SMMA = smoothed moving average (similar to EMA but different smoothing)
    
    def smma(series, period):
        """Smoothed Moving Average"""
        sma = np.full_like(series, np.nan, dtype=float)
        if len(series) < period:
            return sma
        # First value is simple SMA
        sma[period-1] = np.mean(series[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT_VALUE) / PERIOD
        for i in range(period, len(series)):
            sma[i] = (sma[i-1] * (period-1) + series[i]) / period
        return sma
    
    # Calculate the three lines
    jaw_raw = smma(close_1d, 13)   # Blue line
    teeth_raw = smma(close_1d, 8)  # Red line
    lips_raw = smma(close_1d, 5)   # Green line
    
    # Apply forward shifts (as per Williams Alligator definition)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    # Shift jaw forward by 8 bars
    if len(jaw) > 8:
        jaw[8:] = jaw_raw[:-8]
    # Shift teeth forward by 5 bars
    if len(teeth) > 5:
        teeth[5:] = teeth_raw[:-5]
    # Shift lips forward by 3 bars
    if len(lips) > 3:
        lips[3:] = lips_raw[:-3]
    
    # 1d volume spike filter (20-period average)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = prices['volume'].iloc[i]
        vol_ma = vol_ma_aligned[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (green > red > blue) + volume spike + price > Teeth
            if lips_val > teeth_val and teeth_val > jaw_val and vol_spike and price > teeth_val:
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips < Teeth < Jaw (green < red < blue) + volume spike + price < Teeth
            elif lips_val < teeth_val and teeth_val < jaw_val and vol_spike and price < teeth_val:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: lines intertwine (no clear trend) or volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when lips <= teeth OR teeth <= jaw (trend weakening) OR volume dries up
                if lips_val <= teeth_val or teeth_val <= jaw_val or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when lips >= teeth OR teeth >= jaw (trend weakening) OR volume dries up
                if lips_val >= teeth_val or teeth_val >= jaw_val or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Williams_Alligator_Volume_Spike"
timeframe = "12h"
leverage = 1.0