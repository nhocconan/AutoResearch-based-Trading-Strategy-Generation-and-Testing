#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and chop regime filter
# - Long: Price breaks above Camarilla H3 (1d) + 1d volume > 2.0x 20-period MA + 1d chop < 61.8 (trending regime)
# - Short: Price breaks below Camarilla L3 (1d) + 1d volume > 2.0x 20-period MA + 1d chop < 61.8
# - Exit: Price returns to Camarilla H4/L4 levels OR chop > 61.8 (range regime)
# - Position sizing: 0.25 discrete level
# - Targets ~12-37 trades/year on 12h timeframe. Uses Camarilla structure from 1d for institutional levels,
#   volume spike confirms participation, chop filter avoids whipsaws in ranging markets.
#   Works in bull/bear: breakouts capture strong moves, chop filter adapts to regime.

name = "12h_1d_camarilla_volume_chop_v2"
timeframe = "12h"
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
    # H4 = close + 1.5*(high-low), H3 = close + 1.25*(high-low), L3 = close - 1.25*(high-low), L4 = close - 1.5*(high-low)
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_h3 = close_1d + 1.25 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.25 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 1d volume MA(20) for spike detection
    volume_ma_20_1d = pd.Series(volume_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sumTR14 / (hh14 - ll14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop_1d = np.where(range_14 > 0, 100 * np.log10(sum_tr_14 / range_14) / np.log10(14), 50)
    # Handle cases where sum_tr_14 is 0
    chop_1d = np.where(np.isnan(chop_1d) | np.isinf(chop_1d), 50, chop_1d)
    
    chop_ma_10_1d = pd.Series(chop_1d).ewm(span=10, min_periods=10, adjust=False).mean().values
    chop_ma_10_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_ma_10_1d)
    
    # Calculate 12h volume MA(20) for entry confirmation
    volume_ma_20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_ma_20_1d_aligned[i]) or np.isnan(chop_ma_10_1d_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 20-period MA
        vol_confirm_12h = volume[i] > volume_ma_20[i]
        
        # 1d volume spike: current volume > 2.0x 20-period MA
        vol_spike_1d = volume_1d[i // 16] > 2.0 * volume_ma_20_1d_aligned[i] if i // 16 < len(volume_1d) else False
        # Better: use aligned volume data
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_spike_1d = vol_1d_current[i] > 2.0 * volume_ma_20_1d_aligned[i]
        
        # Chop regime: CHOP < 61.8 = trending regime (favor breakouts)
        chop_regime = chop_ma_10_1d_aligned[i] < 61.8
        
        if position == 0:  # Flat - look for Camarilla breakouts
            # Long entry: Price breaks above H3 + vol confirm + vol spike + chop regime
            if (close[i] > camarilla_h3_aligned[i] and vol_confirm_12h and 
                vol_spike_1d and chop_regime):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L3 + vol confirm + vol spike + chop regime
            elif (close[i] < camarilla_l3_aligned[i] and vol_confirm_12h and 
                  vol_spike_1d and chop_regime):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to H4/L4 levels OR chop > 61.8 (range regime)
            if position == 1:  # Long position
                if close[i] <= camarilla_h4_aligned[i] or chop_ma_10_1d_aligned[i] >= 61.8:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] >= camarilla_l4_aligned[i] or chop_ma_10_1d_aligned[i] >= 61.8:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals