#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 12h trend filter + volume confirmation
# - Uses 12h EMA(13,8,5) smoothed for trend direction (long when jaw < teeth < lips)
# - Uses 6h Williams Alligator (jaw=13, teeth=8, lips=5) for entry signals
# - Requires volume > 1.3 * 24-period volume average for confirmation
# - Fixed position size 0.25 to manage drawdown and reduce fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in bull markets via Alligator alignment + breakouts, in bear via proper trend filtering

name = "6h_12h_alligator_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(13,8,5) for trend filter (Alligator components)
    close_12h = df_12h['close'].values
    jaw_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth_12h = pd.Series(close_12h).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips_12h = pd.Series(close_12h).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Pre-compute 6h Williams Alligator
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    typical = (high + low + close) / 3.0
    
    jaw = pd.Series(typical).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(typical).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(typical).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Pre-compute ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume confirmation: volume > 1.3 * 24-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or np.isnan(lips_12h_aligned[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(atr[i]) or atr[i] <= 0 or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine 12h trend direction from Alligator alignment
        uptrend_12h = jaw_12h_aligned[i] < teeth_12h_aligned[i] < lips_12h_aligned[i]
        downtrend_12h = jaw_12h_aligned[i] > teeth_12h_aligned[i] > lips_12h_aligned[i]
        
        # Determine 6h Alligator alignment
        alligator_long = jaw[i] < teeth[i] < lips[i]
        alligator_short = jaw[i] > teeth[i] > lips[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # Exit conditions: stoploss or trend reversal
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif not (uptrend_12h and alligator_long):  # Exit if trend or Alligator fails
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
            elif not (downtrend_12h and alligator_short):  # Exit if trend or Alligator fails
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for entries in direction of 12h trend with 6h Alligator alignment and volume confirmation
            if uptrend_12h and alligator_long and volume_confirm[i]:
                position = 1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = 0.25
            elif downtrend_12h and alligator_short and volume_confirm[i]:
                position = -1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = -0.25
    
    return signals