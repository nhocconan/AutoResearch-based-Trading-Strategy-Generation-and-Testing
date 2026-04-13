#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla L3/H3 breakout with 1d volume spike and 1d chop regime filter
    # Long: price breaks above H3 + volume > 1.5x 20-period average + chop > 61.8 (range) → mean reversion long
    # Short: price breaks below L3 + volume > 1.5x 20-period average + chop > 61.8 (range) → mean reversion short
    # Uses discrete sizing (0.25) to minimize fee drag and ATR-based stoploss
    # Target: 12-30 trades/year to stay within 12h optimal range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots, volume confirmation, and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    hl_range = high_1d - low_1d
    h3 = pivot + 1.1 * hl_range
    l3 = pivot - 1.1 * hl_range
    
    # Calculate 1d volume average (20-period) for confirmation
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Chopiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(TR,14) / (max(HH,14) - min(LL,14))) / log10(14)
    tr_1d = np.maximum(
        high_1d - low_1d,
        np.maximum(
            np.abs(high_1d - np.roll(close_1d, 1)),
            np.abs(low_1d - np.roll(close_1d, 1))
        )
    )
    tr_1d[0] = high_1d[0] - low_1d[0]  # First TR
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    max_hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    denominator = max_hh_14 - min_ll_14
    chop = np.where(
        denominator > 0,
        100 * np.log10(sum_tr_14 / denominator) / np.log10(14),
        50.0  # Neutral when denominator=0
    )
    chop_filter = chop > 61.8  # Range regime (mean reversion favorable)
    
    # Align all indicators to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    # Calculate ATR using true range approximation for 12h timeframe
    atr_12h = np.zeros(n)
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        if i < 14:
            atr_12h[i] = tr  # Simple average for warmup
        else:
            atr_12h[i] = 0.93 * atr_12h[i-1] + 0.07 * tr  # Wilder's smoothing
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        vol_avg_20_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_avg_20_12h[i]):
            signals[i] = 0.0
            continue
        volume_confirmed = volume[i] > 1.5 * vol_avg_20_12h[i]
        
        # Breakout conditions: price breaks Camarilla levels with volume and chop filter
        breakout_long = (close[i] > h3_aligned[i]) and volume_confirmed and chop_aligned[i]
        breakout_short = (close[i] < l3_aligned[i]) and volume_confirmed and chop_aligned[i]
        
        # Stoploss: 2x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.0 * atr_12h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.0 * atr_12h[i]
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "12h_1d_camarilla_chop_volume_v1"
timeframe = "12h"
leverage = 1.0