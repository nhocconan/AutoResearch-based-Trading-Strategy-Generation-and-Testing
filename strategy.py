#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + volume confirmation + 1d trend filter.
# Uses Bill Williams Alligator (Jaw/Teeth/Lips) to identify trend direction.
# Long when Lips > Teeth > Jaw + volume spike + price > 1d EMA50.
# Short when Lips < Teeth < Jaw + volume spike + price < 1d EMA50.
# Exit when Alligator lines re-interlace (Lips crosses Teeth) or volume drops.
# Designed to catch strong trends in both bull and bear markets with low trade frequency.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator on 4h timeframe
    # Jaw (Blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (Red): 8-period SMMA, shifted 5 bars forward
    # Lips (Green): 5-period SMMA, shifted 3 bars forward
    close = prices['close'].values
    
    # Calculate SMMA (Smoothed Moving Average)
    def smma(series, period):
        sma = np.full_like(series, np.nan, dtype=float)
        if len(series) >= period:
            sma[period-1] = np.mean(series[:period])
            for i in range(period, len(series)):
                sma[i] = (sma[i-1] * (period-1) + series[i]) / period
        return sma
    
    smma_13 = smma(close, 13)
    smma_8 = smma(close, 8)
    smma_5 = smma(close, 5)
    
    # Shift as per Alligator specification
    jaw = np.roll(smma_13, 8)   # Jaw: 13-period SMMA shifted 8 bars
    teeth = np.roll(smma_8, 5)  # Teeth: 8-period SMMA shifted 5 bars
    lips = np.roll(smma_5, 3)   # Lips: 5-period SMMA shifted 3 bars
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema50 = ema50_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) + volume spike + price > EMA50
            if lips_val > teeth_val and teeth_val > jaw_val and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips < Teeth < Jaw (bearish alignment) + volume spike + price < EMA50
            elif lips_val < teeth_val and teeth_val < jaw_val and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Alligator lines re-interlace (Lips crosses Teeth) or volume drops
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Lips crosses below Teeth (loss of bullish alignment)
                if lips_val < teeth_val or vol < 0.7 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Lips crosses above Teeth (loss of bearish alignment)
                if lips_val > teeth_val or vol < 0.7 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Williams_Alligator_Volume_EMA50_Trend"
timeframe = "4h"
leverage = 1.0