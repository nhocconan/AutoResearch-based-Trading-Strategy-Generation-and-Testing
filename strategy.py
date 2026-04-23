#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator alignment with 1d EMA50 trend filter and volume confirmation.
Long when Alligator jaws < teeth < lips (bullish alignment) AND price > teeth AND 1d EMA50 rising AND volume > 1.8x 20-period average.
Short when Alligator jaws > teeth > lips (bearish alignment) AND price < teeth AND 1d EMA50 falling AND volume > 1.8x 20-period average.
Exit when Alligator alignment breaks or price crosses teeth.
Uses 1d HTF for EMA50 trend (avoids whipsaws in ranging markets). Target: 50-150 total trades over 4 years (12-37/year).
Williams Alligator (SMAs: jaws=13*8, teeth=8*8, lips=5*8) captures trend strength and alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: SMAs with specific periods
    # Jaws: Blue line - 13-period SMMA shifted 8 bars ahead
    # Teeth: Red line - 8-period SMMA shifted 5 bars ahead  
    # Lips: Green line - 5-period SMMA shifted 3 bars ahead
    def smma(source, period):
        """Smoothed Moving Average"""
        if len(source) < period:
            return np.full_like(source, np.nan, dtype=float)
        result = np.full_like(source, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaws = smma(close, 13)  # 13-period SMMA
    teeth = smma(close, 8)   # 8-period SMMA
    lips = smma(close, 5)    # 5-period SMMA
    
    # Shift as per Alligator definition (jaw shifted 8, teeth 5, lips 3)
    jaws_shifted = np.roll(jaws, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # First shifted values are invalid
    jaws_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13+8, 8+5, 5+3, 50, 20)  # Alligator components, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaws_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        jaw = jaws_shifted[i]
        tooth = teeth_shifted[i]
        lip = lips_shifted[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Alligator alignment conditions
        bullish_alignment = jaw < tooth < lip  # Jaws < Teeth < Lips (green > red > blue from below)
        bearish_alignment = jaw > tooth > lip  # Jaws > Teeth > Lips (green < red < blue from above)
        
        if position == 0:
            # Long: Bullish alignment AND price > teeth AND EMA50 rising AND volume spike
            if bullish_alignment and price > tooth and ema_rising and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment AND price < teeth AND EMA50 falling AND volume spike
            elif bearish_alignment and price < tooth and ema_falling and volume[i] > 1.8 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Alligator bullish alignment breaks OR price crosses below teeth
                if not bullish_alignment or price < tooth:
                    exit_signal = True
            elif position == -1:
                # Short exit: Alligator bearish alignment breaks OR price crosses above teeth
                if not bearish_alignment or price > tooth:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Williams_Alligator_Alignment_1dEMA50_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0