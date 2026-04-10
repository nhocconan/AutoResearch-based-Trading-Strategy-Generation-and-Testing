#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 1d choppiness regime filter
# - Donchian(20) breakout captures strong momentum moves
# - Volume > 1.5x 20-bar average confirms breakout strength
# - 1d Choppiness Index > 61.8 indicates ranging market (fade breakouts)
# - 1d Choppiness Index < 38.2 indicates trending market (follow breakouts)
# - Long when price breaks above Donchian(20) high + volume spike + CHOP < 38.2
# - Short when price breaks below Donchian(20) low + volume spike + CHOP < 38.2
# - Exit when price returns to Donchian(20) middle or opposite breakout occurs
# - Uses discrete position sizing (0.30) to balance return and risk
# - Targets ~25-35 trades/year (100-140 total over 4 years) to avoid fee drag
# - Works in both bull (follow trends) and bear (fade false breakouts in chop) markets

name = "4h_1d_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period) on 4h data
    lookback = 20
    highest_high = pd.Series(prices['high'].values).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(prices['low'].values).rolling(window=lookback, min_periods=lookback).min().values
    donchian_middle = (highest_high + lowest_low) / 2
    
    # Pre-compute 4h volume confirmation: > 1.5x 20-period average
    volume_20_avg = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], 0])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Max High - Min Low over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Choppiness Index = 100 * log10(tr_sum / range_14) / log10(14)
    chop = np.where(
        (range_14 > 0) & (tr_sum > 0),
        100 * np.log10(tr_sum / range_14) / np.log10(14),
        50  # Default when range is zero
    )
    
    # Align 1d Chop to 4h timeframe (with completed bar delay)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_20_avg[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: breakout above Donchian high + volume spike + trending market (CHOP < 38.2)
            if (prices['close'].iloc[i] > highest_high[i] and 
                vol_spike.iloc[i] and 
                chop_aligned[i] < 38.2):
                position = 1
                signals[i] = 0.30
            # Short signal: breakout below Donchian low + volume spike + trending market (CHOP < 38.2)
            elif (prices['close'].iloc[i] < lowest_low[i] and 
                  vol_spike.iloc[i] and 
                  chop_aligned[i] < 38.2):
                position = -1
                signals[i] = -0.30
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price returns to Donchian middle
            # 2. Opposite breakout occurs
            if position == 1:  # Long position
                if (prices['close'].iloc[i] <= donchian_middle[i] or 
                    prices['close'].iloc[i] < lowest_low[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30  # Hold long
            elif position == -1:  # Short position
                if (prices['close'].iloc[i] >= donchian_middle[i] or 
                    prices['close'].iloc[i] > highest_high[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30  # Hold short
    
    return signals