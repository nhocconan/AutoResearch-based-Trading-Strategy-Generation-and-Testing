# Solution
#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 12h Williams Alligator (3 SMAs) with 1d trend filter and volume confirmation.
In uptrend (price > 1d EMA50), go long when Alligator lines are bullish aligned (jaw < teeth < lips) and price > lips.
In downtrend (price < 1d EMA50), go short when Alligator lines are bearish aligned (jaw > teeth > lips) and price < lips.
Uses volume confirmation (>1.5x 20-period average) to filter weak signals.
Williams Alligator helps identify trend strength and avoids whipsaws in ranging markets.
Target: 20-50 trades/year (80-200 total over 4 years) to avoid excessive fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Williams Alligator: Jaw (13-period SMA, shifted 8), Teeth (8-period SMA, shifted 5), Lips (5-period SMA, shifted 3)
    close_12h = df_12h['close'].values
    jaw_raw = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values
    teeth_raw = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values
    lips_raw = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values
    
    # Apply shifts (Alligator specific)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    # Set invalid values for shifted periods
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        trend = ema_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 1.5  # Volume spike filter
        
        if position == 0:
            # Enter long: bullish Alligator alignment + uptrend + price above lips + volume spike
            if (jaw_val < teeth_val and teeth_val < lips_val and  # Bullish alignment
                price_close > lips_val and 
                price_close > trend and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish Alligator alignment + downtrend + price below lips + volume spike
            elif (jaw_val > teeth_val and teeth_val > lips_val and  # Bearish alignment
                  price_close < lips_val and 
                  price_close < trend and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Alligator lines cross (trend weakening) or price crosses opposite lip
            if position == 1:
                if jaw_val > teeth_val or price_close < lips_val:  # Loss of bullish structure
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if jaw_val < teeth_val or price_close > lips_val:  # Loss of bearish structure
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_12h_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0