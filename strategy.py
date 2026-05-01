#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and choppiness regime filter
# Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-period average AND 1d chop > 61.8 (range regime)
# Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-period average AND 1d chop > 61.8 (range regime)
# Exit when price returns to Donchian(20) midpoint OR chop < 38.2 (trend regime)
# Designed for low frequency (50-150 trades over 4 years) with clear structure and volume confirmation

name = "12h_Donchian20_1dVolumeChop_Breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for regime and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = vol_1d / np.where(vol_ma_20 > 0, vol_ma_20, 1)  # Avoid division by zero
    
    # 1d choppiness index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR(14) - sum of TR over 14 periods
    atr_14 = np.full_like(tr, np.nan)
    for i in range(14, len(tr)):
        if not np.isnan(tr[i-13:i+1]).any():
            atr_14[i] = np.nansum(tr[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    hh_14 = np.full_like(high_1d, np.nan)
    ll_14 = np.full_like(low_1d, np.nan)
    for i in range(14, len(high_1d)):
        hh_14[i] = np.nanmax(high_1d[i-13:i+1])
        ll_14[i] = np.nanmin(low_1d[i-13:i+1])
    
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    chop = np.full_like(close_1d, np.nan)
    for i in range(14, len(close_1d)):
        if not np.isnan(atr_14[i]) and hh_14[i] > ll_14[i]:
            chop[i] = 100 * np.log10(atr_14[i] / (hh_14[i] - ll_14[i])) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # 12h Donchian channels (20-period)
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(donchian_window, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(chop_aligned[i]) or np.isnan(vol_ratio_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        # Regime and volume filters
        range_regime = chop_aligned[i] > 61.8  # Chop > 61.8 = ranging market
        volume_spike = vol_ratio_aligned[i] > 1.5  # Volume > 1.5x average
        
        if position == 0:  # Flat - look for new entries
            # Only trade in range regime with volume spike
            if range_regime and volume_spike:
                # Long: price breaks above Donchian high
                if close[i] > highest_high[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian low
                elif close[i] < lowest_low[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # No volume spike or not ranging - stay flat
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price returns to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                exit_long = True
            # Exit when market trends (chop < 38.2) - avoid false breakouts in trends
            elif chop_aligned[i] < 38.2:
                exit_long = True
            
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when price returns to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                exit_short = True
            # Exit when market trends (chop < 38.2) - avoid false breakouts in trends
            elif chop_aligned[i] < 38.2:
                exit_short = True
            
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals