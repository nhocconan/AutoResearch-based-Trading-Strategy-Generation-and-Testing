#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter with 1d trend filter and volume confirmation.
# Long when 12h Choppiness > 61.8 (ranging) AND price > 1d EMA34 AND volume > 2x 20-period average.
# Short when 12h Choppiness > 61.8 (ranging) AND price < 1d EMA34 AND volume > 2x 20-period average.
# Exit when Choppiness < 38.2 (trending) or price crosses EMA34 in opposite direction.
# This strategy mean-reverts in ranging markets (Choppy) and avoids trending regimes.
# Works in both bull and bear by adapting to market regime via Choppiness Index.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.

name = "12h_Choppiness_1dEMA34_Volume_MeanReversion"
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
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 12h data for Choppiness Index calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 12h Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=14).sum()
    highest_high = df_12h['high'].rolling(window=14, min_periods=14).max()
    lowest_low = df_12h['low'].rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr14 / (highest_high - lowest_low)) / np.log10(14)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop_values)
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 14)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_aligned[i]
        price_above_ema = close[i] > ema34_1d_aligned[i]
        price_below_ema = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Enter long in ranging market when price above EMA34
            long_cond = (chop_val > 61.8) and price_above_ema and volume_filter[i]
            # Enter short in ranging market when price below EMA34
            short_cond = (chop_val > 61.8) and price_below_ema and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when market trends or price crosses below EMA34
            if chop_val < 38.2 or price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when market trends or price crosses above EMA34
            if chop_val < 38.2 or price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals