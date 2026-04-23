#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator alignment with 1d trend filter and volume confirmation.
Long when Alligator jaws (blue) < teeth (red) < lips (green) AND 1d EMA50 rising AND volume > 1.8x 20-period average.
Short when Alligator jaws > teeth > lips AND 1d EMA50 falling AND volume > 1.8x 20-period average.
Exit when Alligator lines crossover (jaws/teeth/lips lose alignment) or volume drops below average.
Uses 1d HTF for EMA50 trend to avoid whipsaws in ranging markets. Target: 75-150 total trades over 4 years (19-37/year).
Williams Alligator uses SMAs of median price (HL/2) with periods 13/8/5 and offsets 8/5/3.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate median price (HL/2) for Alligator
    median_price = (high + low) / 2.0
    
    # Williams Alligator: 3 SMAs of median price
    # Jaw (blue): 13-period SMA, shifted 8 bars
    # Teeth (red): 8-period SMA, shifted 5 bars
    # Lips (green): 5-period SMA, shifted 3 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    # Alligator needs max(13+8, 8+5, 5+3) = 21, plus EMA50 (50), volume MA (20)
    start_idx = max(21, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment conditions
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Bullish alignment: jaw < teeth < lips (Alligator opening up)
        bullish_aligned = jaw_val < teeth_val and teeth_val < lips_val
        # Bearish alignment: jaw > teeth > lips (Alligator opening down)
        bearish_aligned = jaw_val > teeth_val and teeth_val > lips_val
        
        price = close[i]
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
        
        if position == 0:
            # Long: Bullish Alligator alignment AND EMA50 rising AND volume spike
            if bullish_aligned and ema_rising and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment AND EMA50 falling AND volume spike
            elif bearish_aligned and ema_falling and volume[i] > 1.8 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Alligator loses bullish alignment OR EMA50 starts falling OR volume drops
                if not bullish_aligned or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]) or volume[i] < vol_ma_val:
                    exit_signal = True
            elif position == -1:
                # Short exit: Alligator loses bearish alignment OR EMA50 starts rising OR volume drops
                if not bearish_aligned or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]) or volume[i] < vol_ma_val:
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