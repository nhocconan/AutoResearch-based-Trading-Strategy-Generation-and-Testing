#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d chop regime filter.
# Long when: price breaks above Donchian(20) high AND volume > 1.5x volume MA(20) AND 1d chop > 61.8 (range)
# Short when: price breaks below Donchian(20) low AND volume > 1.5x volume MA(20) AND 1d chop > 61.8
# Exit when price crosses Donchian(20) midline.
# Designed for 4h timeframe with low trade frequency (target: 20-40/year) to avoid fee drag.
# Uses 4h for price action, volume for confirmation, and 1d chop to avoid trending markets where breakouts fail.
# Works in bull markets via breakouts in uptrend, in bear markets via breakdowns in downtrend.
# Chop filter avoids false breakouts in strong trends and focuses on range-bound breakout/mean reversion.
name = "4h_Donchian20_Volume_ChopFilter"
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
    
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: volume > 1.5x volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # 1d Chop index for regime filter (higher = more range-bound)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 days
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop index: 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    denominator = hh_1d - ll_1d
    chop = np.where(denominator > 0, 100 * np.log10(atr_sum / denominator) / np.log10(14), 50)
    chop[denominator <= 0] = 50  # avoid division by zero or negative
    
    chop_range = chop > 61.8  # range-bound regime
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_range)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Sufficient warmup for Donchian(20) and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high + volume confirm + chop range
            long_condition = (close[i] > donchian_high[i]) and volume_confirm[i] and chop_aligned[i]
            # Short: break below Donchian low + volume confirm + chop range
            short_condition = (close[i] < donchian_low[i]) and volume_confirm[i] and chop_aligned[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: cross below Donchian midline
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: cross above Donchian midline
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals