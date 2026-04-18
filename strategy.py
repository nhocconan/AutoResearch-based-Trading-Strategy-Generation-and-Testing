# Solution
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price crosses 12h Supertrend (ATR=10, mult=3) with volume confirmation.
# Supertrend identifies trend direction using ATR-based bands.
# Price above Supertrend = uptrend, below = downtrend.
# Volume spike (>2x 20-period average) confirms conviction.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
name = "6h_Supertrend_12h_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Supertrend
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Supertrend on 12h data
    high_12h = pd.Series(df_12h['high'].values)
    low_12h = pd.Series(df_12h['low'].values)
    close_12h = pd.Series(df_12h['close'].values)
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = abs(high_12h - close_12h.shift(1))
    tr3 = abs(low_12h - close_12h.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr.rolling(window=10, min_periods=10).mean()
    
    # Supertrend calculation
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + (3 * atr_12h)
    lower_band = hl2 - (3 * atr_12h)
    
    # Initialize Supertrend
    supertrend = np.full(len(close_12h), np.nan)
    uptrend = np.full(len(close_12h), True)
    
    for i in range(10, len(close_12h)):
        # Upper band logic
        if close_12h.iloc[i-1] <= upper_band.iloc[i-1]:
            upper_band.iloc[i] = min(upper_band.iloc[i], upper_band.iloc[i-1])
        else:
            upper_band.iloc[i] = upper_band.iloc[i]
        
        # Lower band logic
        if close_12h.iloc[i-1] >= lower_band.iloc[i-1]:
            lower_band.iloc[i] = max(lower_band.iloc[i], lower_band.iloc[i-1])
        else:
            lower_band.iloc[i] = lower_band.iloc[i]
        
        # Supertrend logic
        if close_12h.iloc[i] <= upper_band.iloc[i]:
            supertrend[i] = upper_band.iloc[i]
            uptrend[i] = False
        elif close_12h.iloc[i] >= lower_band.iloc[i]:
            supertrend[i] = lower_band.iloc[i]
            uptrend[i] = True
        else:
            supertrend[i] = supertrend[i-1]
            uptrend[i] = uptrend[i-1]
    
    # Align Supertrend and uptrend to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    uptrend_aligned = align_htf_to_ltf(prices, df_12h, uptrend.astype(float))
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(supertrend_aligned[i]) or np.isnan(uptrend_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        st_value = supertrend_aligned[i]
        is_uptrend = uptrend_aligned[i] > 0.5  # Convert back to boolean
        
        if position == 0:
            # Long: Price above Supertrend AND uptrend AND volume spike
            if price > st_value and is_uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below Supertrend AND downtrend AND volume spike
            elif price < st_value and not is_uptrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below Supertrend
            if price < st_value:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above Supertrend
            if price > st_value:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals