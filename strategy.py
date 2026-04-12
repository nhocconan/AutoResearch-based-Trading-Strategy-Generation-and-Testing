# 12h_1d_williams_alligator_v1
# Hypothesis: Use Williams Alligator (13,8,5 SMAs) on 1d timeframe to determine trend direction and phase (sleeping/awakening/feeding).
# Enter long when price is above Alligator's lips (13 SMA) and jaws-teeth-lips are properly aligned (bullish alignment: jaws < teeth < lips).
# Enter short when price is below lips and bearish alignment (jaws > teeth > lips).
# Add volume confirmation (volume > 1.5x 20-period average) to avoid false signals.
# Exit on opposite alignment or price crossing the teeth (8 SMA).
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drift.
# Williams Alligator works in both bull and bear markets by identifying trending vs ranging phases.

name = "12h_1d_williams_alligator_v1"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: SMAs of median price (typical price = (H+L+C)/3)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Jaws: 13-period SMMA, teeth: 8-period, lips: 5-period
    # Using SMA as approximation (SMMA would require Wilder's smoothing)
    jaws_1d = pd.Series(typical_price_1d).rolling(window=13, min_periods=13).mean().values
    teeth_1d = pd.Series(typical_price_1d).rolling(window=8, min_periods=8).mean().values
    lips_1d = pd.Series(typical_price_1d).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator lines to 12h timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Bullish alignment: jaws < teeth < lips (Alligator waking up to eat)
        bullish_align = jaws_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i]
        # Bearish alignment: jaws > teeth > lips (Alligator waking up to hunt)
        bearish_align = jaws_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]
        
        # Long entry: price above lips, bullish alignment, volume confirmation
        if (close[i] > lips_aligned[i] and bullish_align and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price below lips, bearish alignment, volume confirmation
        elif (close[i] < lips_aligned[i] and bearish_align and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: opposite alignment or price crosses teeth
        elif position == 1 and (not bullish_align or close[i] < teeth_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not bearish_align or close[i] > teeth_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals