#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike filter and choppiness regime
# - Long: Price breaks above Camarilla H3 level + 1d volume > 1.5x 20-period MA + Chop(14) < 38.2 (trending regime)
# - Short: Price breaks below Camarilla L3 level + 1d volume > 1.5x 20-period MA + Chop(14) < 38.2
# - Exit: Price returns to Camarilla pivot point (mean reversion) OR chop regime ends (Chop > 61.8)
# - Position sizing: 0.25 discrete level
# - Camarilla levels from 1d provide institutional support/resistance, volume spike confirms participation,
#   chop filter ensures we only trade in trending markets where breakouts work. Targets ~40-80 trades/year.

name = "4h_1d_camarilla_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # H4 = Pivot + 1.1*Range/2, H3 = Pivot + 1.1*Range/4, L3 = Pivot - 1.1*Range/4, L4 = Pivot - 1.1*Range/2
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    pivot_1d = typical_price_1d
    camarilla_h3 = pivot_1d + 1.1 * range_1d / 4.0
    camarilla_l3 = pivot_1d - 1.1 * range_1d / 4.0
    camarilla_pivot = pivot_1d
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Calculate 1d volume confirmation: current volume > 1.5x 20-period MA
    volume_ma_20_1d = pd.Series(volume_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
    vol_spike = vol_1d_current > 1.5 * volume_ma_20_1d_aligned
    
    # Calculate choppiness index regime filter (14-period)
    # Chop = 100 * log10(sum(TR,14) / (max(HH,14) - min(LL,14))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denominator = highest_high_14 - lowest_low_14
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # avoid div by zero
    chop = 100 * np.log10(atr_14 / chop_denominator) / np.log10(14)
    chop = np.where(np.isnan(chop), 50.0, chop)  # neutral when undefined
    
    # Chop < 38.2 = trending regime (favor breakouts)
    chop_trending = chop < 38.2
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_spike[i]) or 
            np.isnan(chop_trending[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for Camarilla breakouts
            # Long entry: Price breaks above H3 + volume spike + trending chop
            if close[i] > camarilla_h3_aligned[i] and vol_spike[i] and chop_trending[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L3 + volume spike + trending chop
            elif close[i] < camarilla_l3_aligned[i] and vol_spike[i] and chop_trending[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to pivot point OR chop regime ends (range-bound)
            if position == 1:  # Long position
                if close[i] <= camarilla_pivot_aligned[i] or not chop_trending[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] >= camarilla_pivot_aligned[i] or not chop_trending[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals