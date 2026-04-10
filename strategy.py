#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume regime + ATR trailing stop
# - Uses Williams Alligator (jaw/teeth/lips) on 12h for trend direction and strength
# - 1d volume regime filter: only trade when volume > 1.5x 20-day average (avoid low-volume chop)
# - ATR-based trailing stop: exit when price moves against position by 2.5x ATR(14)
# - Designed for 12h timeframe to capture medium-term swings in both bull and bear markets
# - Alligator stays flat in ranging markets, reducing false signals
# - Volume filter ensures trades occur during sufficient participation
# - ATR stop adapts to volatility, tightening in low vol, widening in high vol
# - Target: 15-25 trades/year on 12h (60-100 total over 4 years) to minimize fee drag

name = "12h_1d_alligator_volume_atrstop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d volume regime filter
    volume_20_avg = df_1d['volume'].rolling(window=20, min_periods=20).mean().values
    vol_regime = df_1d['volume'] > (1.5 * volume_20_avg)
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # Pre-compute ATR(14) for trailing stop
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 12h Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    median_price_12h = (df_12h['high'] + df_12h['low']) / 2
    jaw = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_regime_aligned[i]) or 
            np.isnan(atr[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator signals: lips > teeth > jaw = uptrend, lips < teeth < jaw = downtrend
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                vol_regime_aligned[i]):
                position = 1
                entry_price = prices['close'].iloc[i]
                highest_since_entry = entry_price
                signals[i] = 0.25
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  vol_regime_aligned[i]):
                position = -1
                entry_price = prices['close'].iloc[i]
                lowest_since_entry = entry_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - manage trailing stop and exit
            # Update highest/lowest since entry
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                # Trailing stop: exit if price drops 2.5*ATR from highest
                if prices['close'].iloc[i] < highest_since_entry - 2.5 * atr[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # Trailing stop: exit if price rises 2.5*ATR from lowest
                if prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals