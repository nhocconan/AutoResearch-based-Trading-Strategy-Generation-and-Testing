#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and chop regime filter
# - Long when price breaks above Donchian(20) high AND 1d volume > 2.0x 20-day average AND chop > 61.8 (range)
# - Short when price breaks below Donchian(20) low AND 1d volume > 2.0x 20-day average AND chop > 61.8 (range)
# - Exit when price touches Donchian(20) midpoint or opposite band
# - Uses discrete position sizing (0.30) to balance reward and risk
# - Targets ~20-40 trades/year (80-160 total over 4 years) to avoid fee drag
# - Donchian provides objective breakout levels, volume confirms institutional participation
# - Chop > 61.8 ensures we trade breakouts from consolidation (high probability continuations)
# - Works in bull markets (breakouts continuation) and bear markets (breakdown continuations)

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Donchian high and low
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Pre-compute 1d volume confirmation: > 2.0x 20-day average
    volume_1d = df_1d['volume'].values
    vol_20_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * vol_20_avg_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute 1d Chopiness Index (14-period)
    # Chop = 100 * log10(sum(ATR(14)) / log10(highest high - lowest low over 14 periods))
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with close_1d index
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = highest_high_14 - lowest_low_14
    
    # Avoid division by zero
    chop_raw = np.where(range_14 > 0, 
                        100 * np.log10(sum_atr14) / np.log10(range_14), 
                        50.0)  # Neutral when range is zero
    chop_1d = pd.Series(chop_raw).ewm(span=14, adjust=False, min_periods=14).mean().values
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        # Regime filter: only trade in ranging markets (chop > 61.8)
        in_range = chop_1d_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long signal: break above Donchian high AND volume spike AND ranging market
            if (close[i] > donchian_high[i] and 
                vol_spike_1d_aligned[i] and 
                in_range):
                position = 1
                signals[i] = 0.30
            # Short signal: break below Donchian low AND volume spike AND ranging market
            elif (close[i] < donchian_low[i] and 
                  vol_spike_1d_aligned[i] and 
                  in_range):
                position = -1
                signals[i] = -0.30
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price touches Donchian midpoint (mean reversion)
            # 2. Price touches opposite Donchian band (failed breakout)
            if position == 1:
                if close[i] <= donchian_mid[i] or close[i] >= donchian_high[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30  # Hold long
            elif position == -1:
                if close[i] >= donchian_mid[i] or close[i] <= donchian_low[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30  # Hold short
    
    return signals