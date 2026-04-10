#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume spike and chop regime filter
# - Long: Price breaks above 1d Donchian upper (20) + 1w volume > 2.0x 20-period MA + 1w chop < 61.8 (trending)
# - Short: Price breaks below 1d Donchian lower (20) + 1w volume > 2.0x 20-period MA + 1w chop < 61.8
# - Exit: Price returns to 1d Donchian midpoint OR chop > 61.8 (range regime)
# - Position sizing: 0.25 discrete level
# - Targets ~15-25 trades/year on 1d timeframe. Uses Donchian structure for breakouts,
#   volume spike confirms institutional participation, chop filter avoids whipsaws in ranging markets.
#   Works in bull/bear: breakouts capture strong moves, chop filter adapts to regime.

name = "1d_1w_donchian_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1w volume MA(20) for spike detection
    volume_ma_20_1w = pd.Series(volume_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    # Calculate 1w Choppiness Index (CHOP) for regime filter
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]
    tr2[0] = np.abs(high_1w[0] - close_1w[0])
    tr3[0] = np.abs(low_1w[0] - close_1w[0])
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sumTR14 / (hh14 - ll14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop_1w = np.where(range_14 > 0, 100 * np.log10(sum_tr_14 / range_14) / np.log10(14), 50)
    # Handle cases where sum_tr_14 is 0
    chop_1w = np.where(np.isnan(chop_1w) | np.isinf(chop_1w), 50, chop_1w)
    
    chop_ma_10_1w = pd.Series(chop_1w).ewm(span=10, min_periods=10, adjust=False).mean().values
    chop_ma_10_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_ma_10_1w)
    
    # Calculate 1d Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2.0
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(volume_ma_20_1w_aligned[i]) or 
            np.isnan(chop_ma_10_1w_aligned[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 20-period MA
        volume_ma_20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
        vol_confirm_1d = volume[i] > volume_ma_20[i]
        
        # 1w volume spike: current volume > 2.0x 20-period MA
        vol_1w_current = align_htf_to_ltf(prices, df_1w, volume_1w)
        vol_spike_1w = vol_1w_current[i] > 2.0 * volume_ma_20_1w_aligned[i]
        
        # Chop regime: CHOP < 61.8 = trending regime (favor breakouts)
        chop_regime = chop_ma_10_1w_aligned[i] < 61.8
        
        if position == 0:  # Flat - look for Donchian breakouts
            # Long entry: Price breaks above upper + vol confirm + vol spike + chop regime
            if (close[i] > highest_high_20[i] and vol_confirm_1d and 
                vol_spike_1w and chop_regime):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below lower + vol confirm + vol spike + chop regime
            elif (close[i] < lowest_low_20[i] and vol_confirm_1d and 
                  vol_spike_1w and chop_regime):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to midpoint OR chop > 61.8 (range regime)
            if position == 1:  # Long position
                if close[i] <= donchian_mid[i] or chop_ma_10_1w_aligned[i] >= 61.8:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] >= donchian_mid[i] or chop_ma_10_1w_aligned[i] >= 61.8:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals