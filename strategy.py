#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Williams Alligator + 12h Trend + Volume Confirmation
# Hypothesis: Williams Alligator (three SMAs) identifies market phases; 
# when aligned with 12h trend and volume confirmation, it captures 
# strong trends while avoiding whipsaws. Works in bull via jaw-teeth-lips 
# alignment up, in bear via alignment down, and avoids ranges via 
# intertwined lines. Target: 20-40 trades/year.
name = "4h_williams_alligator_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 4h: Jaw(13,8), Teeth(8,5), Lips(5,3)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Get 12h trend: EMA(21)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    trend_12h = pd.Series(df_12h['close'].values).ewm(span=21, adjust=False).mean().values
    trend_12h_4h = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Volume confirmation: 4h volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(trend_12h_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: Alligator lines intertwine (no clear trend) or trend turns bearish
            if not ((jaws_up := jaw[i] > teeth[i] > lips[i]) or 
                    (jaws_down := jaw[i] < teeth[i] < lips[i])) or \
               close[i] < trend_12h_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: Alligator lines intertwine or trend turns bullish
            if not ((jaws_up := jaw[i] > teeth[i] > lips[i]) or 
                    (jaws_down := jaw[i] < teeth[i] < lips[i])) or \
               close[i] > trend_12h_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: Alligator aligned up (JAW > TEETH > LIPS) with volume and bullish 12h trend
            if jaw[i] > teeth[i] > lips[i] and vol_confirm and close[i] > trend_12h_4h[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: Alligator aligned down (JAW < TEETH < LIPS) with volume and bearish 12h trend
            elif jaw[i] < teeth[i] < lips[i] and vol_confirm and close[i] < trend_12h_4h[i]:
                position = -1
                signals[i] = -0.25
    
    return signals