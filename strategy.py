#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness Index regime filter + 1-day Donchian breakout with volume confirmation
# Choppiness Index identifies ranging vs trending markets. In trending regimes (CHOP < 38.2),
# we take Donchian breakouts with volume confirmation. In ranging regimes (CHOP > 61.8), we fade
# the extremes. This adapts to both bull and bear markets by following the trend when strong
# and mean-reverting when choppy. Target: 20-40 trades/year for low fee decay.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1-day Choppiness Index (14-period) for regime detection ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of True Range over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(atr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(14)
    chop = np.where(np.isnan(chop), 50, chop)  # default to middle if undefined
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 1-day Donchian Channel (20-period) ===
    donch_high_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    donch_high_20_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20_1d)
    donch_low_20_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20_1d)
    
    # === 1-day Volume Spike (vs 20-period average) ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 12-hour price for execution ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(donch_high_20_1d_aligned[i]) or
            np.isnan(donch_low_20_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(close_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        chop_val = chop_aligned[i]
        vol_spike = volume_1d[i] > vol_ma_20_1d_aligned[i] * 1.5  # use 1d volume for confirmation
        
        # Regime-based logic
        if chop_val < 38.2:  # Trending regime
            if position == 0 and vol_spike:
                if close_12h_aligned[i] > donch_high_20_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close_12h_aligned[i] < donch_low_20_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Exit long if price returns to midpoint or volatility drops
                midpoint = (donch_high_20_1d_aligned[i] + donch_low_20_1d_aligned[i]) / 2
                if close_12h_aligned[i] < midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short if price returns to midpoint
                midpoint = (donch_high_20_1d_aligned[i] + donch_low_20_1d_aligned[i]) / 2
                if close_12h_aligned[i] > midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
                    
        elif chop_val > 61.8:  # Ranging regime - fade the extremes
            if position == 0 and vol_spike:
                if close_12h_aligned[i] > donch_high_20_1d_aligned[i]:
                    signals[i] = -0.25  # fade the breakout - go short
                    position = -1
                elif close_12h_aligned[i] < donch_low_20_1d_aligned[i]:
                    signals[i] = 0.25   # fade the breakout - go long
                    position = 1
            elif position == 1:
                # Exit long if price returns to lower band or moves to upper band
                if close_12h_aligned[i] < donch_low_20_1d_aligned[i] or close_12h_aligned[i] > donch_high_20_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short if price returns to upper band or moves to lower band
                if close_12h_aligned[i] > donch_high_20_1d_aligned[i] or close_12h_aligned[i] < donch_low_20_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:  # Neutral regime - no position
            signals[i] = 0.0
            position = 0
    
    return signals

name = "12h_ChopRegime_Donchian20_1dVolumeFade"
timeframe = "12h"
leverage = 1.0