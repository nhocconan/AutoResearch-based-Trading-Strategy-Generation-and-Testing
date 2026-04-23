#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Williams Alligator (Jaw/Teeth/Lips) with volume confirmation and ATR trailing stop.
Long when price > Alligator Lips AND Lips > Teeth AND Teeth > Jaw (bullish alignment) AND volume > 1.3x 20-period average.
Short when price < Alligator Lips AND Lips < Teeth AND Teeth < Jaw (bearish alignment) AND volume > 1.3x 20-period average.
Exit when price crosses Alligator Teeth or ATR trailing stop hit (2.5*ATR from extreme since entry).
Uses discrete position sizing (0.25) to minimize fee drag and manage drawdown.
Designed for 4h timeframe targeting ~20-30 trades/year per symbol (80-120 total over 4 years).
Williams Alligator is a trend-following indicator that works well in both trending and ranging markets when combined with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1h Williams Alligator components (for smoother alignment to 4h)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 13:
        return np.zeros(n)
    
    # Alligator components using SMMA (Smoothed Moving Average)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (df_1h['high'] + df_1h['low']) / 2.0
    
    def smma(series, period):
        """Smoothed Moving Average"""
        sma = series.rolling(window=period, min_periods=period).mean()
        result = np.full_like(series.values, np.nan, dtype=float)
        if len(sma) >= period:
            result[period-1] = sma.iloc[period-1]
            for i in range(period, len(series)):
                result[i] = (result[i-1] * (period-1) + series.iloc[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Shift components as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted periods
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Align 1h Alligator to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1h, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1h, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1h, lips_shifted)
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13)  # vol MA needs 20, Alligator needs 13+8=21 for jaw shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        if position == 0:
            # Long: Bullish Alligator alignment AND volume spike
            if (price > lips_val and lips_val > teeth_val and teeth_val > jaw_val and 
                volume[i] > 1.3 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Bearish Alligator alignment AND volume spike
            elif (price < lips_val and lips_val < teeth_val and teeth_val < jaw_val and 
                  volume[i] > 1.3 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price crosses Alligator Teeth (trend reversal signal)
            if position == 1 and price < teeth_val:
                exit_signal = True
            elif position == -1 and price > teeth_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsAlligator_1h_VolumeSpike_ATRTrailingStop_TeethExit"
timeframe = "4h"
leverage = 1.0