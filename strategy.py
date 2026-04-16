#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d Trend Filter and Volume Confirmation
# Uses Alligator (Jaw, Teeth, Lips) on 12h for trend direction, filtered by 1d price above/below 200 EMA
# and volume > 1.5x average. Works in both bull and bear markets by following higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === 1d data (higher timeframe for trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === Williams Alligator on 12h ===
    # Jaw (Blue line): 13-period SMMA smoothed 8 periods ahead
    # Teeth (Red line): 8-period SMMA smoothed 5 periods ahead
    # Lips (Green line): 5-period SMMA smoothed 3 periods ahead
    
    # Calculate SMMA (Smoothed Moving Average)
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        smma_arr = np.full_like(arr, np.nan, dtype=float)
        smma_arr[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            smma_arr[i] = (smma_arr[i-1] * (period-1) + arr[i]) / period
        return smma_arr
    
    jaw = smma(close_12h, 13)
    teeth = smma(close_12h, 8)
    lips = smma(close_12h, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # === 1d EMA200 for trend filter ===
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === 12h volume spike detection ===
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > (1.5 * vol_ma_20_12h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or
            np.isnan(lips[i]) or
            np.isnan(ema200_1d_aligned[i]) or
            np.isnan(vol_ma_20_12h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema200 = ema200_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            atr_12h = np.abs(high_12h - low_12h)
            atr_ma = pd.Series(atr_12h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_12h, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_12h = np.abs(high_12h - low_12h)
            atr_ma = pd.Series(atr_12h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_12h, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when Alligator lines cross in bearish order (Lips < Teeth < Jaw)
            if lips_val < teeth_val < jaw_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when Alligator lines cross in bullish order (Lips > Teeth > Jaw)
            if lips_val > teeth_val > jaw_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume spike and price on correct side of 1d EMA200
            if vol_spike_val:
                # Go long when price above 1d EMA200 and bullish Alligator alignment (Lips > Teeth > Jaw)
                if price > ema200 and lips_val > teeth_val > jaw_val:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short when price below 1d EMA200 and bearish Alligator alignment (Lips < Teeth < Jaw)
                elif price < ema200 and lips_val < teeth_val < jaw_val:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Alligator_1dEMA200_Volume_Filter"
timeframe = "12h"
leverage = 1.0