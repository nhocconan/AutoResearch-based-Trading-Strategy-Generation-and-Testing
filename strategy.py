#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band breakout with 1d volume regime filter and ATR trailing stop
# - Uses 4h Bollinger Bands (20, 2.0) for breakout signals (long at upper band, short at lower band)
# - Confirms with 1d volume regime: volume > 1.5x 20-day average (high participation environment)
# - Uses ATR(14) trailing stop: exits when price retraces 2.0x ATR from extreme
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Bollinger Bands work in ranging markets (mean reversion at bands) and trending markets (breakout)
# - Volume regime filter ensures breakouts occur during high conviction moves, reducing false signals
# - ATR stop adapts to volatility, keeping risk consistent across market regimes
# - Target: 30-60 trades/year on 4h timeframe (120-240 total over 4 years)

name = "4h_1d_bb_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d volume > 1.5x 20-period average (volume regime filter)
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_regime_1d = volume_1d > (1.5 * avg_volume_20)
    
    # Align 1d volume regime to 4h
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime_1d)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Bollinger Bands (20, 2.0)
    bb_period = 20
    bb_std = 2.0
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma_20 + (bb_std * bb_std_dev)
    lower_band = sma_20 - (bb_std * bb_std_dev)
    
    # 4h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(sma_20[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(volume_regime_aligned[i]) or np.isnan(atr_14[i]) or
            atr_14[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high
            if low[i] <= highest_since_entry - (2.0 * atr_14[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low
            if high[i] >= lowest_since_entry + (2.0 * atr_14[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Bollinger Band breakout with volume regime confirmation
            if (high[i] >= upper_band[i] and    # Break above upper band
                volume_regime_aligned[i]):      # Volume regime confirmation
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = high[i]
                signals[i] = 0.25
            elif (low[i] <= lower_band[i] and    # Break below lower band
                  volume_regime_aligned[i]):     # Volume regime confirmation
                position = -1
                highest_since_entry = low[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals