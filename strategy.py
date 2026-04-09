#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h EMA trend filter and volume confirmation
# - Williams %R(14) for mean reversion signals: long when < -80, short when > -20
# - 12h EMA(50) trend filter: only take longs when price > EMA, shorts when price < EMA
# - Volume confirmation: 6h volume > 1.5x 20-period average to avoid false signals
# - ATR(14) trailing stop at 2.0x ATR from extreme for risk control
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: ~12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Novelty: Williams %R is underutilized; combining with 12h EMA filter adapts to trend
# - Works in both bull/bear: mean reversion in ranges, EMA filter prevents counter-trend in strong trends

name = "6h_12h_williams_r_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 6h timeframe (completed 12h bar only)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # 6h volume > 1.5x 20-period average (volume confirmation)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
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
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(atr[i]) or
            atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high OR Williams %R > -20 (overbought)
            if low[i] <= highest_since_entry - (2.0 * atr[i]) or williams_r[i] > -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low OR Williams %R < -80 (oversold)
            if high[i] >= lowest_since_entry + (2.0 * atr[i]) or williams_r[i] < -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R signals with EMA trend filter and volume confirmation
            # Long: Williams %R < -80 (oversold) AND price > 12h EMA AND volume spike
            if williams_r[i] < -80 and close[i] > ema_50_12h_aligned[i] and volume_spike[i]:
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = 0.25
            # Short: Williams %R > -20 (overbought) AND price < 12h EMA AND volume spike
            elif williams_r[i] > -20 and close[i] < ema_50_12h_aligned[i] and volume_spike[i]:
                position = -1
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals