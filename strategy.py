#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + 1w Donchian breakout with volume confirmation
# Choppiness Index > 61.8 = ranging market (mean revert at Donchian bands)
# Choppiness Index < 38.2 = trending market (follow Donchian breakouts)
# Uses 1w Donchian(20) for structure, 1w for Choppiness regime, volume spike for confirmation
# Designed to work in both bull (trend following) and bear (mean reversion in ranges)
# Target: 50-150 total trades over 4 years (~12-37/year) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian and Choppiness calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian channels (20-period)
    donchian_high = np.full(len(df_1w), np.nan)
    donchian_low = np.full(len(df_1w), np.nan)
    for i in range(19, len(df_1w)):
        donchian_high[i] = np.max(high_1w[i-19:i+1])
        donchian_low[i] = np.min(low_1w[i-19:i+1])
    
    # Calculate 1w Choppiness Index (14-period)
    atr_1w = np.zeros(len(df_1w))
    tr_1w = np.zeros(len(df_1w))
    for i in range(1, len(df_1w)):
        tr = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
        tr_1w[i] = tr
    
    # Calculate ATR(14)
    for i in range(len(df_1w)):
        if i < 14:
            atr_1w[i] = np.nan
        else:
            atr_1w[i] = np.mean(tr_1w[i-13:i+1])
    
    # Calculate Choppiness Index
    chop = np.full(len(df_1w), np.nan)
    for i in range(14, len(df_1w)):
        sum_tr = np.sum(tr_1w[i-13:i+1])
        hh = np.max(high_1w[i-13:i+1])
        ll = np.min(low_1w[i-13:i+1])
        if hh != ll and sum_tr > 0:
            chop[i] = 100 * np.log10(sum_tr / (hh - ll)) / np.log10(14)
        else:
            chop[i] = 50.0  # neutral when range is zero
    
    # Align indicators to 12h timeframe (wait for weekly close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Volume filter: volume > 1.5 x 8-period average (4 days of 12h bars)
    vol_ma_8 = np.full(n, np.nan)
    for i in range(7, n):
        vol_ma_8[i] = np.mean(volume[i-7:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly data (20 bars), ATR (14), chop (14), volume MA (8)
    start_idx = max(20, 14, 8)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_8[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_8[i]
        chop_val = chop_aligned[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Regime filters
        ranging_market = chop_val > 61.8  # Chop > 61.8 = ranging
        trending_market = chop_val < 38.2  # Chop < 38.2 = trending
        
        if position == 0:
            # In ranging market: mean reversion at Donchian bands
            if ranging_market and vol_filter:
                if price <= lower:  # Near lower band -> long
                    signals[i] = size
                    position = 1
                elif price >= upper:  # Near upper band -> short
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            # In trending market: follow breakouts
            elif trending_market and vol_filter:
                if price > upper:  # Break above -> long
                    signals[i] = size
                    position = 1
                elif price < lower:  # Break below -> short
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to opposite band or regime changes
            if price >= upper or chop_val > 61.8:  # Return to upper band or market ranges
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to opposite band or regime changes
            if price <= lower or chop_val > 61.8:  # Return to lower band or market ranges
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Choppiness_Donchian_Breakout_Volume"
timeframe = "12h"
leverage = 1.0