#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume spike confirmation
# Long when Alligator jaws (13) < teeth (8) < lips (5) AND close > 1d EMA34 AND volume > 2.0x 20-bar avg
# Short when Alligator jaws (13) > teeth (8) > lips (5) AND close < 1d EMA34 AND volume > 2.0x 20-bar avg
# Exit when Alligator lines cross (trend change)
# Uses discrete position sizing (0.25) to balance capture and risk.
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years) to avoid overtrading.
# Williams Alligator identifies trending vs ranging markets via smoothed moving averages.
# Volume spike confirms participation, reducing false signals.
# 1d EMA34 trend filter ensures alignment with higher timeframe direction, working in both bull and bear regimes.

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter and Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator (SMMA with specific periods)
    # Williams Alligator uses Smoothed Moving Average (SMMA) with specific shifts
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    
    def smma(source, period):
        """Smoothed Moving Average"""
        smma_vals = np.full_like(source, np.nan, dtype=float)
        if len(source) < period:
            return smma_vals
        # First value is simple SMA
        smma_vals[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(source)):
            smma_vals[i] = (smma_vals[i-1] * (period-1) + source[i]) / period
        return smma_vals
    
    # Calculate SMMA components
    jaw_raw = smma(close_1d, 13)
    teeth_raw = smma(close_1d, 8)
    lips_raw = smma(close_1d, 5)
    
    # Apply shifts (Williams Alligator specific)
    jaw = np.roll(jaw_raw, 8)  # shifted 8 bars forward
    teeth = np.roll(teeth_raw, 5)  # shifted 5 bars forward
    lips = np.roll(lips_raw, 3)  # shifted 3 bars forward
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: >2.0x 20-bar average volume (tight to avoid overtrading)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 13)  # volume MA, EMA34, and Alligator warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_jaw = jaw_aligned[i]
        curr_teeth = teeth_aligned[i]
        curr_lips = lips_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Alligator lines cross (jaws > teeth) indicating trend change
            if curr_jaw > curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator lines cross (jaws < teeth) indicating trend change
            if curr_jaw < curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Alligator aligned: jaws < teeth < lips = bullish alignment
            # Alligator reversed: jaws > teeth > lips = bearish alignment
            bullish_aligned = curr_jaw < curr_teeth and curr_teeth < curr_lips
            bearish_aligned = curr_jaw > curr_teeth and curr_teeth > curr_lips
            
            # Long when bullish alignment AND close > 1d EMA34 AND volume confirmation
            if bullish_aligned and curr_close > curr_ema34_1d and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when bearish alignment AND close < 1d EMA34 AND volume confirmation
            elif bearish_aligned and curr_close < curr_ema34_1d and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals