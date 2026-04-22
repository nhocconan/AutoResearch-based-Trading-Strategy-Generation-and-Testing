#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 12h trend filter and volume confirmation.
# Williams Alligator uses smoothed medians (Jaw: 13-period SMMA, Teeth: 8-period, Lips: 5-period).
# When the three lines are intertwined (market sleeping), we avoid trades.
# When they diverge (market waking up) with alignment (Lips > Teeth > Jaw for long, reverse for short),
# we enter in the direction of the 12h EMA trend with volume confirmation (>1.5x 20-period average volume).
# Designed for low trade frequency (~15-30/year) to minimize fee decay.
# Works in both bull and bear markets by following higher timeframe trend and requiring clear trend alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for Williams Alligator (once before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate median price for each period
    median_12h = (high_12h + low_12h) / 2
    
    # Williams Alligator: Smoothed Medians (SMMA)
    # Jaw: 13-period SMMA of median, Teeth: 8-period, Lips: 5-period
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_12h, 13)
    teeth = smma(median_12h, 8)
    lips = smma(median_12h, 5)
    
    # Calculate 50-period EMA on 12h close for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 4h timeframe (waits for 12h bar to close)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_val = ema_50_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_confirm = vol > 1.5 * vol_ma
        
        # Alligator alignment: Lips > Teeth > Jaw for bullish, Lips < Teeth < Jaw for bearish
        bullish_alignment = lips_val > teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val < jaw_val
        
        if position == 0:
            # Long conditions: bullish alignment + price above EMA + volume confirmation
            if bullish_alignment and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish alignment + price below EMA + volume confirmation
            elif bearish_alignment and price < ema_val and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: when Alligator starts to sleep (lines intertwine) or trend breaks
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when alignment breaks down or price crosses below Teeth
                if not bullish_alignment or price < teeth_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when alignment breaks down or price crosses above Teeth
                if not bearish_alignment or price > teeth_val:
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