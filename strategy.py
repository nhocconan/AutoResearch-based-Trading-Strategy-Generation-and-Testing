#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA trend filter + volume confirmation
# - Williams %R(14) on 6h: long when < -80 (oversold), short when > -20 (overbought)
# - Trend filter: 1d EMA(50) - price must be above EMA for longs, below for shorts
# - Volume confirmation: 6h volume > 1.3x 20-period average to avoid low-volume false signals
# - ATR-based stop: exit when price moves 2.0x ATR against position (trailing via highest/lowest since entry)
# - Position size: 0.25 (discrete level to minimize fee churn)
# - Target: ~50-150 total trades over 4 years (12-37/year) per 6h strategy guidelines
# - Works in both bull/bear: Williams %R captures mean reversion in ranges, EMA filter ensures trades with trend
# - Novelty: Williams %R is underutilized; combining with 1d EMA and volume avoids whipsaws in strong trends

name = "6h_1d_williamsr_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)  # avoid div by zero
    
    # 6h volume > 1.3x 20-period average (volume confirmation)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * avg_volume_20)
    
    # 6h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(atr[i]) or
            atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high OR Williams %R > -50 (exit oversold)
            if low[i] <= highest_since_entry - (2.0 * atr[i]) or \
               williams_r[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low OR Williams %R < -50 (exit overbought)
            if high[i] >= lowest_since_entry + (2.0 * atr[i]) or \
               williams_r[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extreme with volume confirmation and trend filter
            # Long: Williams %R < -80 (oversold) AND price > 1d EMA(50) AND volume spike
            if williams_r[i] < -80 and close[i] > ema_50_1d_aligned[i] and volume_spike[i]:
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = high[i]
                signals[i] = 0.25
            # Short: Williams %R > -20 (overbought) AND price < 1d EMA(50) AND volume spike
            elif williams_r[i] > -20 and close[i] < ema_50_1d_aligned[i] and volume_spike[i]:
                position = -1
                highest_since_entry = low[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals