#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray with volume confirmation
# Long when Alligator jaws < teeth < lips (bullish alignment) AND Elder Ray bull power > 0 AND volume > 1.5x 20-bar avg
# Short when Alligator jaws > teeth > lips (bearish alignment) AND Elder Ray bear power < 0 AND volume > 1.5x 20-bar avg
# Exit when Alligator alignment breaks or Elder Ray power crosses zero
# Uses discrete position sizing (0.25) to minimize fee churn while capturing trend moves.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h.
# Williams Alligator identifies trend direction and alignment; Elder Ray measures bull/bear power behind the move.
# Volume confirmation ensures institutional participation, reducing false signals.
# Works in bull markets (trend continuation) and bear markets (trend following shorts).

name = "12h_WilliamsAlligator_ElderRay_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator (13,8,5 SMAs of median price)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate median price: (high + low) / 2
    median_price = (df_1d['high'].values + df_1d['low'].values) / 2
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 1d data for Elder Ray (EMA13 of high/low)
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Elder Ray: Bull Power = High - EMA13(Close), Bear Power = Low - EMA13(Close)
    ema_13 = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema_13
    bear_power = df_1d['low'].values - ema_13
    
    # Align Elder Ray components to 12h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 20)  # Alligator/Elder Ray warmup + volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Alligator alignment breaks OR bull power <= 0
            if not (jaw_val < teeth_val < lips_val) or bull_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks OR bear power >= 0
            if not (jaw_val > teeth_val > lips_val) or bear_val >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Alligator bullish alignment AND bull power > 0 AND volume confirmation
            if jaw_val < teeth_val < lips_val and bull_val > 0 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Alligator bearish alignment AND bear power < 0 AND volume confirmation
            elif jaw_val > teeth_val > lips_val and bear_val < 0 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals