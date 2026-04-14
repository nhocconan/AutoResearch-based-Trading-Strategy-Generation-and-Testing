#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Choppiness Index regime filter with 1-day Williams %R mean reversion.
# In ranging markets (CHOP > 61.8), Williams %R extremes signal reversals: buy when %R < -80, sell when %R > -20.
# In trending markets (CHOP < 38.2), we avoid trades to prevent whipsaw.
# This strategy targets mean reversion in ranging conditions, which is effective in both bull and bear markets
# when price oscillates within ranges. Volume confirmation filters low-liquidity noise.
# Expected trades: 20-40 per year per symbol (80-160 total over 4 years), within optimal range to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Williams %R and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    
    # Williams %R (14-period) on 1d
    williams_len = 14
    if len(df_1d) < williams_len:
        return np.zeros(n)
    
    highest_high = pd.Series(df_1d['high']).rolling(window=williams_len, min_periods=williams_len).max()
    lowest_low = pd.Series(df_1d['low']).rolling(window=williams_len, min_periods=williams_len).min()
    williams_r = -100 * (highest_high - df_1d['close']) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(50).values  # neutral when range=0
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Choppiness Index (14-period) on 1d
    chop_len = 14
    if len(df_1d) < chop_len:
        return np.zeros(n)
    
    atr_1d = pd.Series(np.sqrt((df_1d['high'] - df_1d['low'])**2))  # True Range approximation without close
    # Proper TR: max(high-low, |high-prev_close|, |low-prev_close|)
    prev_close = df_1d['close'].shift(1)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - prev_close)
    tr3 = np.abs(df_1d['low'] - prev_close)
    atr_1d = pd.Series(np.maximum(tr1, np.maximum(tr2, tr3)))
    atr_sum = atr_1d.rolling(window=chop_len, min_periods=chop_len).sum()
    highest_high_14 = pd.Series(df_1d['high']).rolling(window=chop_len, min_periods=chop_len).max()
    lowest_low_14 = pd.Series(df_1d['low']).rolling(window=chop_len, min_periods=chop_len).min()
    chop = 100 * np.log10(atr_sum / (highest_high_14 - lowest_low_14)) / np.log10(chop_len)
    chop = chop.replace([np.inf, -np.inf], np.nan).fillna(50).values  # neutral when range=0
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: 1.5x average volume (4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, williams_len, chop_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: Choppiness Index > 61.8 = ranging (mean revert), < 38.2 = trending (avoid)
        is_ranging = chop_aligned[i] > 61.8
        is_trending = chop_aligned[i] < 38.2
        
        # Williams %R signals: oversold < -80, overbought > -20
        williams_oversold = williams_r_aligned[i] < -80
        williams_overbought = williams_r_aligned[i] > -20
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: ranging market + Williams %R oversold + volume
            if is_ranging and williams_oversold and volume_confirmed:
                position = 1
                signals[i] = position_size
            # Enter short: ranging market + Williams %R overbought + volume
            elif is_ranging and williams_overbought and volume_confirmed:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral (-50) or regime shifts to trending
            if williams_r_aligned[i] > -50 or is_trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to neutral (-50) or regime shifts to trending
            if williams_r_aligned[i] < -50 or is_trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_WilliamsR_Choppiness_Volume_v1"
timeframe = "4h"
leverage = 1.0