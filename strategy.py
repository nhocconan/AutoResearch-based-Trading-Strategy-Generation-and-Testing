#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and chop regime filter
# - Long when price breaks above Donchian(20) high with 1d volume > 1.5x average and chop < 61.8 (trending)
# - Short when price breaks below Donchian(20) low with 1d volume > 1.5x average and chop < 61.8 (trending)
# - Exit on opposite Donchian(10) break or ATR(14) stoploss (2.0x)
# - Uses 12h timeframe targeting 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
# - 1d volume confirmation ensures breakout strength with participation
# - Chop regime filter avoids false breakouts in ranging markets
# - Discrete position sizing (0.25) to minimize fee churn

name = "12h_1d_donchian_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d volume confirmation: > 1.5x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 1d chop regime filter: chop < 61.8 (trending market)
    # Chop = 100 * log10(sum(atr(14),14) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = np.zeros_like(tr)
    atr_14_1d[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_1d[i] = (atr_14_1d[i-1] * (14-1) + tr[i]) / 14
    
    sum_atr_14 = pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(sum_atr_14 / (max_high_14 - min_low_14 + 1e-10)) / np.log10(14)
    chop_filter_1d = chop_1d < 61.8  # trending regime
    chop_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_filter_1d)
    
    # 12h Donchian channels
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Donchian(20) for entry
    donch_high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Donchian(10) for exit
    donch_high_10 = pd.Series(high_12h).rolling(window=10, min_periods=10).max().values
    donch_low_10 = pd.Series(low_12h).rolling(window=10, min_periods=10).min().values
    
    # 12h ATR(14) for stoploss
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]
    atr_14_12h = np.zeros_like(tr_12h)
    atr_14_12h[14-1] = np.mean(tr_12h[:14])
    for i in range(14, len(tr_12h)):
        atr_14_12h[i] = (atr_14_12h[i-1] * (14-1) + tr_12h[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_high_20[i]) or np.isnan(donch_low_20[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(chop_filter_1d_aligned[i]) or
            np.isnan(atr_14_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Donchian(10) break, ATR stoploss, or chop regime change
            if (close_12h[i] < donch_low_10[i] or 
                close_12h[i] < entry_price - 2.0 * entry_atr or
                chop_filter_1d_aligned[i] == False):  # chop >= 61.8 (ranging)
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Donchian(10) break, ATR stoploss, or chop regime change
            if (close_12h[i] > donch_high_10[i] or 
                close_12h[i] > entry_price + 2.0 * entry_atr or
                chop_filter_1d_aligned[i] == False):  # chop >= 61.8 (ranging)
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian(20) break with volume and chop filters
            if vol_spike_1d_aligned[i] and chop_filter_1d_aligned[i]:
                # Long signal: price breaks above Donchian(20) high
                if close_12h[i] > donch_high_20[i]:
                    position = 1
                    entry_price = close_12h[i]
                    entry_atr = atr_14_12h[i]
                    signals[i] = 0.25
                # Short signal: price breaks below Donchian(20) low
                elif close_12h[i] < donch_low_20[i]:
                    position = -1
                    entry_price = close_12h[i]
                    entry_atr = atr_14_12h[i]
                    signals[i] = -0.25
    
    return signals