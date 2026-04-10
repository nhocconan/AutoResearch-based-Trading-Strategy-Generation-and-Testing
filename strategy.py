#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d volume spike and chop regime filter
# - Williams %R(14) identifies overbought/oversold conditions on 12h
# - Volume spike filter: 1d volume > 1.8x 30-period average (institutional participation)
# - Chop regime filter: 1d Chopiness Index(14) > 61.8 (range-bound market for mean reversion)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.5x ATR(14) on 12h
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Mean reversion profits in ranging markets; volume/chop filters avoid false signals in trends

name = "12h_1d_williamsr_volume_chop_v1"
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
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_30 = pd.Series(volume_1d).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume_1d > (1.8 * avg_volume_30)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Pre-compute 1d Chopiness Index (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index = 100 * log10(tr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    hl_range = hh - ll
    chop = np.where(hl_range > 0, 100 * np.log10(tr_sum / hl_range) / np.log10(14), 50)
    chop_filter = chop > 61.8  # Chop > 61.8 indicates ranging market
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter)
    
    # Pre-compute 12h Williams %R (14)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (highest_high - close) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          ((highest_high - close_12h) / (highest_high - lowest_low)) * -100, 
                          -50)
    
    # Pre-compute 12h ATR(14) for stoploss
    tr_12h1 = high_12h - low_12h
    tr_12h2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr_12h3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr_12h1, np.maximum(tr_12h2, tr_12h3))
    tr_12h[0] = tr_12h1[0]
    atr_14 = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(atr_14[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(chop_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) OR stoploss hit
            if williams_r[i] > -20 or close_12h[i] < entry_price - 2.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) OR stoploss hit
            if williams_r[i] < -80 or close_12h[i] > entry_price + 2.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries with volume and chop filters
            if vol_spike_aligned[i] and chop_filter_aligned[i]:
                # Long: Williams %R < -80 (oversold) in ranging market
                if williams_r[i] < -80:
                    position = 1
                    entry_price = close_12h[i]
                    signals[i] = 0.25
                # Short: Williams %R > -20 (overbought) in ranging market
                elif williams_r[i] > -20:
                    position = -1
                    entry_price = close_12h[i]
                    signals[i] = -0.25
    
    return signals