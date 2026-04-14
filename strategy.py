#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points using prior day's OHLC
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Pivot point: (H + L + C) / 3
    pp = (prev_high + prev_low + prev_close) / 3
    # Resistance and support levels
    r1 = 2 * pp - prev_low
    s1 = 2 * pp - prev_high
    r2 = pp + (high_1d - low_1d)
    s2 = pp - (high_1d - low_1d)
    
    # Align pivot levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Choppiness regime filter: CHOP(14) > 61.8 = range (mean revert), CHOP < 38.2 = trending
    # We use chop to avoid trending markets where pivot breaks fail
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = high_series.rolling(window=14, min_periods=14).max().values
    min_low = low_series.rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr / (max_high - min_low)) / np.log10(14)
    chop = np.where((max_high - min_low) == 0, 50, chop)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for volume and chop calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(avg_vol[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above R2 with volume confirmation AND chop > 61.8 (range market)
            if price > r2_aligned[i] and vol > 1.5 * avg_vol[i] and chop[i] > 61.8:
                position = 1
                signals[i] = position_size
            # Short: price breaks below S2 with volume confirmation AND chop > 61.8 (range market)
            elif price < s2_aligned[i] and vol > 1.5 * avg_vol[i] and chop[i] > 61.8:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below S1 OR chop < 38.2 (trending market - exit range play)
            if price < s1_aligned[i] or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above R1 OR chop < 38.2 (trending market - exit range play)
            if price > r1_aligned[i] or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Pivot_Breakout_ChopFilter"
timeframe = "4h"
leverage = 1.0