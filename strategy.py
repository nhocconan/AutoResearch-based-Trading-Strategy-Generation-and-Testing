#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with Elder Ray power and volume confirmation.
# Uses weekly trend filter and daily volume spike to filter signals.
# Long when Green line > Red line (bullish alignment) + Bull Power > 0 + volume spike.
# Short when Red line > Green line (bearish alignment) + Bear Power > 0 + volume spike.
# Exits when alignment breaks or volume drops.
# Works in trending markets (both bull and bear) by following the Alligator's jaw/teeth/lips.
# Target: 15-25 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter (Alligator based on weekly)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Load daily data for Elder Ray and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Williams Alligator (13,8,5 smoothed with future shift)
    # Jaw (13-period SMMA shifted 8 bars)
    # Teeth (8-period SMMA shifted 5 bars)
    # Lips (5-period SMMA shifted 3 bars)
    def smma(data, period):
        sma = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            sma[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    jaw_raw = smma(close_1w, 13)
    teeth_raw = smma(close_1w, 8)
    lips_raw = smma(close_1w, 5)
    
    # Apply shifts (Jaw: +8, Teeth: +5, Lips: +3)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    # Set invalid values to NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Elder Ray (13-period EMA)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d
    bear_power = ema13_1d - low_1d
    
    # Daily volume spike (20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align to 12h
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = prices['volume'].iloc[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        vol_ma = vol_ma_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        # Alligator conditions
        bullish_alignment = lips_val > teeth_val > jaw_val  # Lips > Teeth > Jaw
        bearish_alignment = jaw_val > teeth_val > lips_val  # Jaw > Teeth > Lips
        
        if position == 0:
            # Long conditions: bullish alignment + bull power > 0 + volume spike
            if bullish_alignment and bull_power_val > 0 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish alignment + bear power > 0 + volume spike
            elif bearish_alignment and bear_power_val > 0 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: alignment breaks or volume drops
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when bullish alignment breaks or volume dries up
                if not bullish_alignment or vol < 0.7 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when bearish alignment breaks or volume dries up
                if not bearish_alignment or vol < 0.7 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_ElderRay_Volume"
timeframe = "12h"
leverage = 1.0