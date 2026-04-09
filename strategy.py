#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily ATR-based volatility breakout with volume confirmation and chop regime filter
# Breakouts from ATR-scaled ranges work in both bull and bear markets when confirmed by volume
# Chop filter (Choppiness Index) avoids false breakouts in sideways markets
# Position size fixed at 0.25 to balance risk and return
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_atr_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR (14-period) for breakout threshold
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d Choppiness Index (14-period) for regime filter
    # Chop = 100 * log10(sum(TR14) / (log10(n) * (max_high - min_low)))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    log_n = np.log10(14)
    chop = 100 * np.log10(atr_sum / (log_n * range_14 + 1e-10))  # Add small epsilon to avoid division by zero
    
    # Align ATR and Chop to 12h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute 12h volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h rolling max/min for breakout channels (20-period)
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(high_ma_20[i]) or np.isnan(low_ma_20[i]) or
            atr_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 12h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Chop regime filter: only trade when market is trending (Chop < 61.8)
        trending_regime = chop_aligned[i] < 61.8
        
        if not (volume_confirmed and trending_regime):
            signals[i] = 0.0
            continue
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit on retracement to 12h VWAP or ATR-based stop
            # Simple exit: close below 12h low MA
            if close[i] < low_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to 12h VWAP or ATR-based stop
            # Simple exit: close above 12h high MA
            if close[i] > high_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # ATR breakout entry with volume and regime confirmation
            # Long breakout: close > 12h high MA + 0.5 * ATR
            # Short breakout: close < 12h low MA - 0.5 * ATR
            if volume_confirmed and trending_regime:
                long_breakout = close[i] > high_ma_20[i] + 0.5 * atr_aligned[i]
                short_breakout = close[i] < low_ma_20[i] - 0.5 * atr_aligned[i]
                
                if long_breakout:
                    position = 1
                    signals[i] = position_size
                elif short_breakout:
                    position = -1
                    signals[i] = -position_size
    
    return signals