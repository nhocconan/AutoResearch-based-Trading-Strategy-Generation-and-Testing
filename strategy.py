#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and choppiness regime filter
# Long when price breaks above 4h Donchian upper band AND 1d volume > 2x 20-period average AND CHOP(14) < 40 (trending)
# Short when price breaks below 4h Donchian lower band AND 1d volume > 2x 20-period average AND CHOP(14) < 40 (trending)
# Exit when price crosses 4h Donchian midpoint (mean reversion) OR CHOP(14) > 60 (range market)
# Uses 4h primary timeframe with 1d HTF for volume and chop filters (more reliable than intraday)
# Donchian channels provide clear breakout levels based on recent price action
# Volume confirmation ensures breakouts have institutional participation
# Chop filter avoids whipsaws in ranging markets (proven edge from 16K+ experiments)
# Discrete sizing (0.30) balances return potential with drawdown control
# Target: 100-200 total trades over 4 years (25-50/year) for 4h timeframe

name = "4h_Donchian20_Breakout_1dVolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for volume and chop filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume spike filter: volume > 2x 20-period average
    vol_1d = df_1d['volume'].values
    if len(vol_1d) >= 20:
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        volume_spike = vol_1d > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(len(df_1d), dtype=bool)
    
    # Calculate 1d choppiness index: CHOP = 100 * log10(sum(TR)/ (HHV(high,14) - LLV(low,14))) / log10(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = np.where((hh_14 - ll_14) != 0, 
                    100 * np.log10(tr_sum / (hh_14 - ll_14)) / np.log10(14), 
                    50)
    
    # Align 1d indicators to 4h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 4h Donchian channels (20-period)
    if len(high) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(volume_spike_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND volume spike AND trending market (CHOP < 40)
            if (close[i] > donchian_upper[i] and 
                volume_spike_aligned[i] and 
                chop_aligned[i] < 40):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below Donchian lower AND volume spike AND trending market (CHOP < 40)
            elif (close[i] < donchian_lower[i] and 
                  volume_spike_aligned[i] and 
                  chop_aligned[i] < 40):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint (mean reversion) OR chop > 60 (range market)
            if close[i] < donchian_mid[i] or chop_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint (mean reversion) OR chop > 60 (range market)
            if close[i] > donchian_mid[i] or chop_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals