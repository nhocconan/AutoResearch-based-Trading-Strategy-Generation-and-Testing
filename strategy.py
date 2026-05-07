# 12h_Williams_Alligator_1dTrend_Volume_Signal
# Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend phases via SMAs of median price.
# Combined with 1-day EMA34 trend filter and volume confirmation to trade in direction of higher timeframe trend.
# Works in bull markets (long when Lips above Teeth above Jaw + uptrend) and bear markets (short when Lips below Teeth below Jaw + downtrend).
# Volume spike filters low-conviction moves. Target: 12-37 trades/year per symbol (50-150 total over 4 years).

name = "12h_Williams_Alligator_1dTrend_Volume_Signal"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Alligator and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Median price for Alligator calculations
    median_price = (high_1d + low_1d) / 2.0
    
    # Williams Alligator: SMAs of median price
    # Jaw: 13-period SMA, shifted 8 bars
    # Teeth: 8-period SMA, shifted 5 bars  
    # Lips: 5-period SMA, shifted 3 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # shift 8 bars forward
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # shift 5 bars forward
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # shift 3 bars forward
    
    # Convert to numpy arrays
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_vals)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_vals)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_vals)
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike detection: 2.0x average volume (24-period = 2 days on 12h chart)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24)  # Ensure we have EMA34 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment), price above EMA34 (uptrend), volume spike
            if (lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment), price below EMA34 (downtrend), volume spike
            elif (lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaw_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Lips cross below Teeth (loss of bullish alignment) OR price crosses below EMA34
            if (lips_aligned[i] < teeth_aligned[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Lips cross above Teeth (loss of bearish alignment) OR price crosses above EMA34
            if (lips_aligned[i] > teeth_aligned[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals