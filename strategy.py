#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d EMA200 trend filter and volume confirmation.
# Williams Alligator uses 3 SMAs (jaw=13, teeth=8, lips=5) to detect trends.
# Long when: Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA200 AND volume > 1.5x 20-period average
# Short when: Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA200 AND volume > 1.5x 20-period average
# Exit when: Alligator alignment breaks (jaws cross teeth or lips)
# The Alligator identifies trends, 1d EMA200 filters for higher timeframe direction, volume confirms strength.
# Works in trending markets by capturing sustained moves. Target: 20-30 trades/year per symbol.
name = "4h_WilliamsAlligator_1dEMA200_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d EMA200 ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 13, 8, 5, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema200 = ema200_1d_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Alligator alignment conditions
        bullish_alignment = lips_val > teeth_val and teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Long entry: Bullish alignment + price > 1d EMA200 + volume spike
            if bullish_alignment and price > ema200 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish alignment + price < 1d EMA200 + volume spike
            elif bearish_alignment and price < ema200 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bullish alignment breaks OR price crosses below 1d EMA200
            if not bullish_alignment or price < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bearish alignment breaks OR price crosses above 1d EMA200
            if not bearish_alignment or price > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals