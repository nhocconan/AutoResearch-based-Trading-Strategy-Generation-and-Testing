#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume confirmation and ATR stoploss
# - Uses 12h Williams Alligator (Jaw=13, Teeth=8, Lips=5) for trend direction
# - Requires 1d volume > 1.5 * 20-period volume average for confirmation
# - Uses ATR(14) for dynamic stoploss (2.5 * ATR) and position sizing (0.25)
# - Alligator convergence/divergence filters whipsaws in ranging markets
# - Works in bull markets via teeth above lips, in bear via teeth below lips
# - Target: 12-30 trades/year on 12h timeframe (50-120 total over 4 years) to avoid fee drag
# - Williams Alligator provides smoothed trend identification that adapts to volatility

name = "12h_1d_williams_alligator_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume confirmation: volume > 1.5 * 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (1.5 * vol_ma_1d)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # Pre-compute 12h Williams Alligator
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    median = (high + low) / 2.0
    
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = pd.Series(median).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # shift forward by 8 periods
    teeth = pd.Series(median).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # shift forward by 5 periods
    lips = pd.Series(median).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # shift forward by 3 periods
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # Pre-compute 12h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_vals[i]) or np.isnan(teeth_vals[i]) or np.isnan(lips_vals[i]) or
            np.isnan(atr[i]) or atr[i] <= 0 or
            np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # Exit conditions: stoploss or trend reversal
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif teeth_vals[i] < lips_vals[i]:  # Trend reversal (teeth below lips)
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # Exit conditions: stoploss or trend reversal
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif teeth_vals[i] > lips_vals[i]:  # Trend reversal (teeth above lips)
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for trend entries with volume confirmation
            if teeth_vals[i] > lips_vals[i] and volume_confirm_aligned[i]:  # Uptrend (teeth above lips)
                position = 1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = 0.25
            elif teeth_vals[i] < lips_vals[i] and volume_confirm_aligned[i]:  # Downtrend (teeth below lips)
                position = -1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = -0.25
    
    return signals