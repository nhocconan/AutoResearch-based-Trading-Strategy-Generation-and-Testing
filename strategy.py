#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band breakout with 1d volume confirmation and chop regime filter
# - Primary signal: Price breaks above/below Bollinger Bands (20, 2.0) on 12h
# - Volume filter: 1d volume > 1.3x 20-period average volume (institutional participation)
# - Chop filter: 1d Choppiness Index > 61.8 (range-bound market) → mean reversion at bands
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.0x ATR(20) on 12h
# - Works in bull/bear: Breakouts capture strong moves; chop filter avoids whipsaws
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines

name = "12h_1d_bb_volume_chop_v1"
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
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.3 * avg_volume_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Pre-compute 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_1 = high_1d - low_1d
    tr_2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr_3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr_1, np.maximum(tr_2, tr_3))
    tr[0] = tr_1[0]
    
    # Sum of True Range over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index = 100 * log10(atr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    hl_range = hh - ll
    chop_raw = np.where(hl_range > 0, atr_sum / hl_range, 1.0)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    chop_filter = chop > 61.8  # Chop > 61.8 = range-bound (mean reversion)
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter)
    
    # Pre-compute 12h Bollinger Bands (20, 2.0)
    close_12h = prices['close'].values
    bb_middle = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    
    # Pre-compute 12h ATR(20) for stoploss
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    
    tr_12h1 = high_12h - low_12h
    tr_12h2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr_12h3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr_12h1, np.maximum(tr_12h2, tr_12h3))
    tr_12h[0] = tr_12h1[0]
    atr_20 = pd.Series(tr_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(chop_filter_aligned[i]) or
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price reverts to middle band OR stoploss hit
            if close_12h[i] < bb_middle[i] or close_12h[i] < entry_price - 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price reverts to middle band OR stoploss hit
            if close_12h[i] > bb_middle[i] or close_12h[i] > entry_price + 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Bollinger Band breakouts with volume and chop filters
            if vol_spike_aligned[i] and chop_filter_aligned[i]:
                # Long: price breaks above upper Bollinger Band
                if close_12h[i] > bb_upper[i]:
                    position = 1
                    entry_price = close_12h[i]
                    signals[i] = 0.25
                # Short: price breaks below lower Bollinger Band
                elif close_12h[i] < bb_lower[i]:
                    position = -1
                    entry_price = close_12h[i]
                    signals[i] = -0.25
    
    return signals