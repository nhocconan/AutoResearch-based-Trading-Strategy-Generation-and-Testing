#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price rejection at 1d VWAP with volume exhaustion filter.
# In both bull and bear markets, price tends to reject at daily VWAP after
# extended moves. Volume exhaustion (declining volume on rejection) confirms
# lack of follow-through. Uses 4h for timing, 1d VWAP as dynamic support/resistance.
# Target: 20-40 trades/year per symbol with strict entry conditions.
name = "4h_VWAP_Rejection_Volume_Exhaustion"
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
    
    # Get 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate typical price and VWAP for each day
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_numerator = (typical_price * df_1d['volume']).cumsum().values
    vwap_denominator = df_1d['volume'].cumsum().values
    vwap = vwap_numerator / vwap_denominator
    # Handle first value (avoid division by zero)
    vwap[0] = typical_price.iloc[0] if hasattr(typical_price, 'iloc') else typical_price[0]
    
    # Align 1d VWAP to 4h
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    
    # Volume exhaustion: current volume < 0.7 * 20-period average (declining volume)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Price change over last 3 periods to detect exhaustion
    price_change_3 = np.zeros_like(close)
    price_change_3[3:] = (close[3:] - close[:-3]) / close[:-3]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 3)  # Ensure volume MA and price change are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(price_change_3[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap_val = vwap_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        pc3 = price_change_3[i]
        
        # Volume exhaustion condition
        volume_exhausted = vol < 0.7 * vol_ma
        
        # Price rejection conditions
        # Long setup: price rejects VWAP from below with bullish momentum exhaustion
        long_setup = (price <= vwap_val * 1.002 and  # Near VWAP (within 0.2%)
                     price > vwap_val * 0.998 and
                     pc3 < 0 and  # Negative momentum over 3 periods
                     volume_exhausted)
        
        # Short setup: price rejects VWAP from above with bearish momentum exhaustion
        short_setup = (price >= vwap_val * 0.998 and  # Near VWAP (within 0.2%)
                      price <= vwap_val * 1.002 and
                      pc3 > 0 and  # Positive momentum over 3 periods
                      volume_exhausted)
        
        if position == 0:
            # Enter long on VWAP rejection from below with volume exhaustion
            if long_setup:
                signals[i] = 0.25
                position = 1
            # Enter short on VWAP rejection from above with volume exhaustion
            elif short_setup:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price moves back above VWAP or volume returns
            if price > vwap_val * 1.005 or vol > vol_ma:  # Price back above VWAP or volume returns
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price moves back below VWAP or volume returns
            if price < vwap_val * 0.995 or vol > vol_ma:  # Price back below VWAP or volume returns
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals