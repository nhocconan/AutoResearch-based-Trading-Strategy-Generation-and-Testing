#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Choppiness Index regime filter + Donchian(20) breakout + volume confirmation.
# Long when: CHOP(14) > 61.8 (range) + price breaks above Donchian upper + volume > 1.5x 20-period avg
# Short when: CHOP(14) > 61.8 (range) + price breaks below Donchian lower + volume > 1.5x 20-period avg
# Exit when: price crosses back through Donchian median (mean of upper/lower)
# Choppiness Index identifies ranging markets where mean reversion works; Donchian provides breakout levels.
# Works in bull (buy range highs) and bear (sell range lows). Target: 20-30 trades/year per symbol.
name = "4h_Choppiness_Range_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1-day data
    # TR = max[(H-L), abs(H-PC), abs(L-PC)]
    tr1 = np.zeros(len(high_1d))
    tr1[0] = high_1d[0] - low_1d[0]  # First TR is just high-low
    for i in range(1, len(high_1d)):
        hl = high_1d[i] - low_1d[i]
        hc = abs(high_1d[i] - close_1d[i-1])
        lc = abs(low_1d[i] - close_1d[i-1])
        tr1[i] = max(hl, hc, lc)
    
    # Calculate ATR(14) for 1-day data
    atr14_1d = np.zeros(len(tr1))
    for i in range(len(tr1)):
        if i < 13:
            atr14_1d[i] = np.nan
        else:
            atr14_1d[i] = np.mean(tr1[i-13:i+1])
    
    # Calculate Choppiness Index: CHOP = 100 * log10(sum(ATR14) / (n * ATR)) / log10(n)
    chop14_1d = np.zeros(len(high_1d))
    n_period = 14
    for i in range(n_period, len(high_1d)):
        if np.isnan(atr14_1d[i]):
            chop14_1d[i] = np.nan
            continue
        sum_atr = np.sum(atr14_1d[i-n_period+1:i+1])
        max_high = np.max(high_1d[i-n_period+1:i+1])
        min_low = np.min(low_1d[i-n_period+1:i+1])
        if max_high == min_low or atr14_1d[i] == 0:
            chop14_1d[i] = 50.0  # Neutral when no range
        else:
            chop14_1d[i] = 100 * np.log10(sum_atr / (n_period * (max_high - min_low))) / np.log10(n_period)
    
    # Calculate Donchian channels on 4H data
    donch_len = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(donch_len-1, n):
        upper[i] = np.max(high[i-donch_len+1:i+1])
        lower[i] = np.min(low[i-donch_len+1:i+1])
    median = (upper + lower) / 2.0
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1D Choppiness Index to 4H timeframe
    chop14_1d_aligned = align_htf_to_ltf(prices, df_1d, chop14_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for Choppiness and Donchian calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(chop14_1d_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(median[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        chop = chop14_1d_aligned[i]
        up = upper[i]
        low = lower[i]
        med = median[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Only trade in ranging markets (CHOP > 61.8 indicates range)
            if chop > 61.8:
                # Long entry: Price breaks above Donchian upper with volume
                if price > up and close[i-1] <= up and vol > 1.5 * vol_ma:
                    signals[i] = 0.25
                    position = 1
                # Short entry: Price breaks below Donchian lower with volume
                elif price < low and close[i-1] >= low and vol > 1.5 * vol_ma:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below Donchian median
            if price < med:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above Donchian median
            if price > med:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals