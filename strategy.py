#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + volume confirmation
# - Uses Williams Alligator (jaw/teeth/lips SMAs) to identify trend direction and strength
# - Confirms with Elder Ray (bull/bear power) to measure trend momentum
# - Requires volume > 1.5x 20-period average for institutional participation
# - Exits when Alligator lines cross (trend weakening) or ATR stoploss (2.0x ATR)
# - Position size: 0.25 (25% of capital) to balance risk and minimize fee drag
# - Target: 12-25 trades/year on 12h timeframe (50-100 total over 4 years)
# - Williams Alligator excels in trending markets (bull/bear) and avoids whipsaws in ranges
# - Elder Ray adds momentum confirmation to avoid false breakouts
# - Works in both bull (teeth above jaw) and bear (teeth below jaw) regimes

name = "12h_alligator_elder_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1w HTF indicators
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Pre-compute 1d HTF indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1w Williams Alligator (jaw=13, teeth=8, lips=5 SMAs of median price)
    median_price_1w = (high_1w + low_1w) / 2
    jaw_1w = pd.Series(median_price_1w).rolling(window=13, min_periods=13).mean().values
    teeth_1w = pd.Series(median_price_1w).rolling(window=8, min_periods=8).mean().values
    lips_1w = pd.Series(median_price_1w).rolling(window=5, min_periods=5).mean().values
    
    # 1d Elder Ray (bull power = high - EMA13, bear power = low - EMA13)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # 1d Volume > 1.5x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    
    # 1d ATR(14) for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF indicators to 12h
    jaw_1w_aligned = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_1w_aligned = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_1w_aligned = align_htf_to_ltf(prices, df_1w, lips_1w)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_1w_aligned[i]) or np.isnan(teeth_1w_aligned[i]) or np.isnan(lips_1w_aligned[i]) or
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: Alligator lines cross (teeth < jaw) or ATR stoploss
            if teeth_1w_aligned[i] < jaw_1w_aligned[i]:  # Alligator sleeping - trend weakening
                position = 0
                signals[i] = 0.0
            elif low[i] <= entry_price - (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Alligator lines cross (teeth > jaw) or ATR stoploss
            if teeth_1w_aligned[i] > jaw_1w_aligned[i]:  # Alligator sleeping - trend weakening
                position = 0
                signals[i] = 0.0
            elif high[i] >= entry_price + (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Alligator alignment with Elder Ray confirmation and volume
            # Strong uptrend: lips > teeth > jaw AND bull power > 0
            # Strong downtrend: lips < teeth < jaw AND bear power < 0
            if (lips_1w_aligned[i] > teeth_1w_aligned[i] > jaw_1w_aligned[i] and  # Aligned up
                bull_power_1d_aligned[i] > 0 and                    # Bull power confirmation
                volume_spike_1d_aligned[i]):                        # Volume confirmation
                position = 1
                entry_price = high[i]
                atr_stop = atr_1d_aligned[i]
                signals[i] = 0.25
            elif (lips_1w_aligned[i] < teeth_1w_aligned[i] < jaw_1w_aligned[i] and  # Aligned down
                  bear_power_1d_aligned[i] < 0 and                   # Bear power confirmation
                  volume_spike_1d_aligned[i]):                       # Volume confirmation
                position = -1
                entry_price = low[i]
                atr_stop = atr_1d_aligned[i]
                signals[i] = -0.25
    
    return signals