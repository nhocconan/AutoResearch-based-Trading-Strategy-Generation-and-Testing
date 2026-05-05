#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + volume spike + 1d chop regime filter for mean reversion in ranging markets
# Long when price breaks above Donchian upper(20) AND volume > 2.0x 20-period average AND chop > 61.8 (range)
# Short when price breaks below Donchian lower(20) AND volume > 2.0x 20-period average AND chop > 61.8 (range)
# Exit when price crosses Donchian middle (mean reversion) OR chop < 38.2 (trending regime)
# Donchian channels provide clear structure, volume confirms participation, chop filter avoids false breakouts in trends
# Target: 12-37 trades/year per symbol (50-150 total over 4 years) for 12h timeframe
# Discrete sizing (0.25) to limit fee drag

name = "12h_Donchian20_VolumeSpike_ChopRegime_MeanReversion"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for chop regime calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Chopiness Index (CHOP) for regime detection
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high_n) - min(low_n))))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(np.maximum(tr1, tr2), tr3)
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align with index
    
    # ATR(14) on 1d
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR(14) over last 14 periods
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Max(high_n) - Min(low_n) over last 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Chopiness Index: CHOP = 100 * log10(sum_atr_14 / (log10(14) * range_14))
    # Avoid division by zero and log of zero/negative
    log10_14 = np.log10(14)
    denominator = log10_14 * range_14
    chop_1d = np.where(
        (denominator > 0) & (sum_atr_14 > 0),
        100 * np.log10(sum_atr_14 / denominator),
        50.0  # neutral value when invalid
    )
    
    # Align 1d chop to 12h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Donchian channels on 12h: upper = max(high,20), lower = min(low,20), middle = (upper+lower)/2
    if len(high) >= 20:
        donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donch_middle = (donch_upper + donch_lower) / 2.0
    else:
        donch_upper = np.full(n, np.nan)
        donch_lower = np.full(n, np.nan)
        donch_middle = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donch_upper[i]) or 
            np.isnan(donch_lower[i]) or 
            np.isnan(donch_middle[i]) or 
            np.isnan(chop_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long/short only in ranging market (CHOP > 61.8) with volume spike
            if chop_1d_aligned[i] > 61.8 and volume_filter[i]:
                # Long: price breaks above Donchian upper
                if close[i] > donch_upper[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian lower
                elif close[i] < donch_lower[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses Donchian middle (mean reversion) OR chop < 38.2 (trending regime)
            if (close[i] < donch_middle[i] or 
                chop_1d_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses Donchian middle (mean reversion) OR chop < 38.2 (trending regime)
            if (close[i] > donch_middle[i] or 
                chop_1d_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals