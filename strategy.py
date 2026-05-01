#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1w volume spike + 1d chop regime filter
# Uses 1w volume confirmation (volume > 1.5x 20-period MA) to validate breakouts
# Uses 1d choppiness index (CHOP > 61.8 = range, CHOP < 38.2 = trend) as regime filter
# In trending regime (CHOP < 38.2): trade Donchian breakouts in direction of trend
# In ranging regime (CHOP > 61.8): fade Donchian breaks (mean reversion at opposite band)
# Designed for low frequency (50-150 trades over 4 years) with clear regime logic

name = "12h_Donchian20_1wVolume_1dChop_Regime_v1"
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
    
    # 1d HTF data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d Choppiness Index calculation
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
    atr_14 = np.zeros_like(tr)
    for i in range(14, len(tr)):
        atr_14[i] = np.nansum(tr[i-13:i+1])
    
    # High-Low range over 14 periods
    max_high_14 = np.full_like(close_1d, np.nan)
    min_low_14 = np.full_like(close_1d, np.nan)
    for i in range(14, len(close_1d)):
        max_high_14[i] = np.nanmax(high_1d[i-13:i+1])
        min_low_14[i] = np.nanmin(low_1d[i-13:i+1])
    
    # Chop = 100 * log10(sum(TR14) / (max(HH14-LL14))) / log10(14)
    range_14 = max_high_14 - min_low_14
    chop = np.full_like(close_1d, np.nan)
    mask = (atr_14 > 0) & (range_14 > 0) & (~np.isnan(range_14))
    chop[mask] = 100 * np.log10(atr_14[mask] / range_14[mask]) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 1w HTF data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w volume confirmation: volume > 1.5x 20-period MA
    vol_1w = df_1w['volume'].values
    vol_ma_20 = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.full_like(vol_1w, np.nan)
    vol_ratio = np.where(vol_ma_20 > 0, vol_1w / vol_ma_20, np.nan)
    vol_spike = vol_ratio > 1.5  # Volume spike confirmation
    vol_spike_aligned = align_htf_to_ltf(prices, df_1w, vol_spike.astype(float), additional_delay_bars=0)
    
    # 12h Donchian(20) channels
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    for i in range(lookback, len(high)):
        highest_high[i] = np.nanmax(high[i-lookback+1:i+1])
        lowest_low[i] = np.nanmin(low[i-lookback+1:i+1])
    
    # Donchian breakout signals
    breakout_up = close > highest_high  # Close above upper band
    breakout_down = close < lowest_low  # Close below lower band
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(lookback, 20, 14)  # Need Donchian, volume MA, and ATR
    
    for i in range(start_idx, n):
        if (np.isnan(chop_aligned[i]) or np.isnan(vol_spike_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters
        trending = chop_aligned[i] < 38.2  # Chop < 38.2 = trending
        ranging = chop_aligned[i] > 61.8   # Chop > 61.8 = ranging
        
        if position == 0:  # Flat - look for new entries
            # Volume confirmation required for all entries
            if vol_spike_aligned[i]:
                # Trending regime: trade breakouts in direction of trend
                if trending:
                    if breakout_up[i]:
                        signals[i] = 0.25
                        position = 1
                    elif breakout_down[i]:
                        signals[i] = -0.25
                        position = -1
                # Ranging regime: fade breakouts (mean reversion)
                elif ranging:
                    if breakout_up[i]:
                        signals[i] = -0.25  # Short at upper band
                        position = -1
                    elif breakout_down[i]:
                        signals[i] = 0.25   # Long at lower band
                        position = 1
            else:
                signals[i] = 0.0  # No volume confirmation - stay flat
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_long = False
            if trending:
                # Exit trending long when price reaches lower Donchian band
                if close[i] <= lowest_low[i]:
                    exit_long = True
            elif ranging:
                # Exit ranging long when price reaches midpoint (mean reversion target)
                midpoint = (highest_high[i] + lowest_low[i]) / 2
                if close[i] >= midpoint:
                    exit_long = True
            else:
                # Transition regime - exit on opposite Donchian touch
                if close[i] <= lowest_low[i]:
                    exit_long = True
            
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            if trending:
                # Exit trending short when price reaches upper Donchian band
                if close[i] >= highest_high[i]:
                    exit_short = True
            elif ranging:
                # Exit ranging short when price reaches midpoint (mean reversion target)
                midpoint = (highest_high[i] + lowest_low[i]) / 2
                if close[i] <= midpoint:
                    exit_short = True
            else:
                # Transition regime - exit on opposite Donchian touch
                if close[i] >= highest_high[i]:
                    exit_short = True
            
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals