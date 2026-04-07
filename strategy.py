#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Choppiness Index + Donchian Breakout + Volume Spike
# Hypothesis: In low volatility (high chop) markets, price tends to mean revert.
# In high volatility (low chop) markets, price trends. We use Donchian breakouts
# in trending regimes (low chop) with volume confirmation to capture strong moves.
# Works in bull markets via upside breakouts, in bear via downside breakouts.
# Choppiness Index filters out ranging markets to reduce false breakouts.
# Target: 20-50 trades/year (80-200 total over 4 years) for 4h timeframe.

name = "4h_chop_donchian_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # ATR(14) for Choppiness Index
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate max/min high/low over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(atr(14)) / (max_high - min_low)) / log10(14)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    denominator = max_high - min_low
    chop = np.where(denominator > 0, 
                    100 * np.log10(atr_sum) / np.log10(14) / np.log10(denominator),
                    50)  # Neutral when denominator is 0
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian Channel (20-period) on 4h
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):
        # Skip if required data not available
        if (np.isnan(chop_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or chop rises above 61.8 (ranging)
            if close[i] < donch_low[i] or chop_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or chop rises above 61.8 (ranging)
            if close[i] > donch_high[i] or chop_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter in trending regime (chop < 38.2) with volume spike
            if vol_ok and chop_aligned[i] < 38.2:
                # Long breakout: price closes above Donchian high
                if close[i] > donch_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below Donchian low
                elif close[i] < donch_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals