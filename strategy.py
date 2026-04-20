#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Chop_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1h and 1d data ONCE before loop
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1h: Trend filter (EMA34) ===
    close_1h = df_1h['close'].values
    ema34_1h = pd.Series(close_1h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1h_aligned = align_htf_to_ltf(prices, df_1h, ema34_1h)
    
    # === 1d: Chop index (14-period) ===
    high_1d = df_1h['high'].values
    low_1d = df_1h['low'].values
    close_1d = df_1h['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first tr
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr14 / (atr14 * 14)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1h, chop)
    
    # === 4h: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        close_val = close[i]
        ema_val = ema34_1h_aligned[i]
        chop_val = chop_1d_aligned[i]
        highest_high_val = highest_high[i]
        lowest_low_val = lowest_low[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_val) or np.isnan(chop_val) or np.isnan(highest_high_val) or np.isnan(lowest_low_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Chop > 61.8 (range) + price breaks above Donchian high + volume
            if (chop_val > 61.8 and
                close_val > highest_high_val and
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Chop > 61.8 (range) + price breaks below Donchian low + volume
            elif (chop_val > 61.8 and
                  close_val < lowest_low_val and
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price re-enters Donchian channel or chop drops (trend)
            if (close_val < highest_high_val or  # re-enter channel
                chop_val < 50):                  # trend mode
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price re-enters Donchian channel or chop drops (trend)
            if (close_val > lowest_low_val or  # re-enter channel
                chop_val < 50):                # trend mode
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals