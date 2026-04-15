#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaws, Teeth, Lips) alignment + volume spike + volatility filter.
# Alligator lines converge in ranging markets (no trade) and diverge in trends (trade).
# Works in bull/bear by catching strong trends after consolidation.
# Target: 12-30 trades/year via strict alignment conditions.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Alligator (based on median price)
    daily = get_htf_data(prices, '1d')
    
    # Williams Alligator: SMAs of median price (HL/2)
    median_price = (daily['high'].values + daily['low'].values) / 2
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values  # 13-period, 8-shift
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values    # 8-period, 5-shift
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values     # 5-period, 3-shift
    
    # Align to 12h timeframe
    jaws_aligned = align_htf_to_ltf(prices, daily, jaws)
    teeth_aligned = align_htf_to_ltf(prices, daily, teeth)
    lips_aligned = align_htf_to_ltf(prices, daily, lips)
    
    # Volatility filter: ATR(14) > 0.5% of price to avoid low volatility chop
    tr1 = daily['high'].values[1:] - daily['low'].values[1:]
    tr2 = np.abs(daily['high'].values[1:] - daily['close'].values[:-1])
    tr3 = np.abs(daily['low'].values[1:] - daily['close'].values[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    vol_filter = atr_14d_aligned > (0.005 * close)
    
    # Volume filter: current volume > 2.0x 20-day average volume
    vol_ma_20d = pd.Series(daily['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20d_aligned = align_htf_to_ltf(prices, daily, vol_ma_20d)
    vol_spike = volume > (2.0 * vol_ma_20d_aligned)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(atr_14d_aligned[i]) or
            np.isnan(vol_ma_20d_aligned[i])):
            continue
        
        # Only trade when volatility is sufficient
        if not vol_filter[i]:
            signals[i] = 0.0
            continue
            
        # Long: Lips > Teeth > Jaws (bullish alignment) + volume spike
        if (lips_aligned[i] > teeth_aligned[i] and 
            teeth_aligned[i] > jaws_aligned[i] and 
            vol_spike[i]):
            signals[i] = 0.25
        
        # Short: Lips < Teeth < Jaws (bearish alignment) + volume spike
        elif (lips_aligned[i] < teeth_aligned[i] and 
              teeth_aligned[i] < jaws_aligned[i] and 
              vol_spike[i]):
            signals[i] = -0.25
        
        # Exit: Alligator lines converge (|Lips - Jaw| < 0.1% of price) or reverse alignment
        elif (np.abs(lips_aligned[i] - jaws_aligned[i]) < 0.001 * close[i]) or \
             (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaws_aligned[i] and signals[i-1] < 0) or \
             (lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaws_aligned[i] and signals[i-1] > 0):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Williams_Alligator_Alignment_Volume_Spike"
timeframe = "12h"
leverage = 1.0