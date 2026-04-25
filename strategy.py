#!/usr/bin/env python3
"""
4h Williams Alligator + Volume Spike + ATR Stoploss
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and exhaustion.
Long when Lips > Teeth > Jaw (bullish alignment) with volume spike and price > Lips.
Short when Lips < Teeth < Jaw (bearish alignment) with volume spike and price < Lips.
Uses ATR-based trailing stop to limit drawdown in bear markets. 4h timeframe targets 20-50 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator: SMAs of median price
    median_price = (high + low) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR for stoploss (20-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for calculations
    start_idx = max(20, 13, 8, 5)  # ATR, jaw, teeth, lips
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        vol_spike = volume_spike[i]
        
        # Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Look for entry signals
            # Long: bullish alignment AND volume spike AND price > lips
            long_entry = bullish_alignment and vol_spike and (curr_close > lips[i])
            # Short: bearish alignment AND volume spike AND price < lips
            short_entry = bearish_alignment and vol_spike and (curr_close < lips[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: bearish alignment OR price < lips (stoploss) OR ATR trailing stop
            atr_stop = curr_close < (entry_price - 2.0 * atr[i])
            if (not bullish_alignment) or (curr_close < lips[i]) or atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: bullish alignment OR price > lips (stoploss) OR ATR trailing stop
            atr_stop = curr_close > (entry_price + 2.0 * atr[i])
            if (not bearish_alignment) or (curr_close > lips[i]) or atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0