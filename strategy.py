#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d volume confirmation + chop regime filter
# - Williams %R(14) on 12h for momentum extremes: > -20 = overbought, < -80 = oversold
# - Volume confirmation: 12h volume > 1.8x 20-period average to filter weak moves
# - Regime filter: 12h Choppiness Index > 61.8 for mean-reversion (fade extremes in ranging markets)
# - ATR(14) trailing stop at 2.0x ATR from extreme for risk control
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: ~12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Novelty: Williams %R is less common than RSI, chop regime prevents false signals in sideways markets
# - Works in both bull/bear: Williams %R adapts to volatility, chop filter avoids whipsaws

name = "12h_williamsr_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d average volume for confirmation
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.8 * avg_volume_1d)
    
    # Align 1d volume spike to 12h timeframe (completed 1d bar only)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # 12h volume > 1.8x 20-period average (volume confirmation)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * avg_volume_20)
    
    # 12h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h Choppiness Index (CHOP) for regime filter
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    chop = np.where(range_14 > 0, 100 * np.log10(atr_14 / range_14) / np.log10(14), 50)
    chop = np.where(np.isnan(chop), 50, chop)
    chop_regime = chop > 61.8  # True when ranging/markets suitable for mean reversion
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(chop_regime[i]) or
            atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high OR Williams %R > -20 (overbought) in chop
            if low[i] <= highest_since_entry - (2.0 * atr[i]) or \
               (chop_regime[i] and williams_r[i] > -20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low OR Williams %R < -80 (oversold) in chop
            if high[i] >= lowest_since_entry + (2.0 * atr[i]) or \
               (chop_regime[i] and williams_r[i] < -80):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extremes with volume confirmation on both timeframes
            # Long: Williams %R < -80 (oversold) AND volume spike on 12h AND 1d
            if williams_r[i] < -80 and volume_spike[i] and volume_spike_1d_aligned[i]:
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = high[i]
                signals[i] = 0.25
            # Short: Williams %R > -20 (overbought) AND volume spike on 12h AND 1d
            elif williams_r[i] > -20 and volume_spike[i] and volume_spike_1d_aligned[i]:
                position = -1
                highest_since_entry = low[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals