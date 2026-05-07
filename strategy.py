#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Choppiness Index regime filter + 1d Williams %R mean reversion.
# In high chop (CHOP > 61.8): mean revert at extreme Williams %R (WR < 20 long, WR > 80 short).
# In low chop (CHOP < 38.2): follow trend (price > SMA50 long, price < SMA50 short).
# Volume confirmation required for all entries.
# This adapts to market regimes: mean reversion in ranging markets, trend following in trending markets.
# Works in both bull and bear by dynamically switching strategies based on market structure.

name = "6h_ChopWilliams_Regime_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Choppiness Index (14-period)
    chop_length = 14
    atr1 = np.abs(high - low)
    tr = np.maximum(np.abs(high - np.roll(low, 1)), np.maximum(np.abs(low - np.roll(close, 1)), atr1))
    tr[0] = atr1[0]  # first TR is just range
    
    # True Range sum over chop_length
    tr_sum = pd.Series(tr).rolling(window=chop_length, min_periods=chop_length).sum().values
    # AT-R = (high-low + |high-previous close| + |low-previous close|)/2 approximation
    # Using actual TR above
    highest_high = pd.Series(high).rolling(window=chop_length, min_periods=chop_length).max().values
    lowest_low = pd.Series(low).rolling(window=chop_length, min_periods=chop_length).min().values
    
    # Avoid division by zero
    range_chop = highest_high - lowest_low
    range_chop = np.where(range_chop == 0, 1e-10, range_chop)
    
    chop = 100 * np.log10(tr_sum / range_chop) / np.log10(chop_length)
    
    # Williams %R from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_wr = highest_high_1d - lowest_low_1d
    range_wr = np.where(range_wr == 0, 1e-10, range_wr)
    
    wr = -100 * (highest_high_1d - close_1d) / range_wr
    wr_aligned = align_htf_to_ltf(prices, df_1d, wr)
    
    # 50-period SMA for trend filter
    sma50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(chop_length, 50, 20)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(chop[i]) or np.isnan(wr_aligned[i]) or np.isnan(sma50[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop[i]
        wr_val = wr_aligned[i]
        
        if position == 0:
            # Determine regime
            if chop_val > 61.8:  # High chop - ranging market
                # Mean reversion at Williams %R extremes
                long_cond = wr_val < 20 and volume_filter[i]
                short_cond = wr_val > 80 and volume_filter[i]
            elif chop_val < 38.2:  # Low chop - trending market
                # Follow trend with price vs SMA50
                long_cond = close[i] > sma50[i] and volume_filter[i]
                short_cond = close[i] < sma50[i] and volume_filter[i]
            else:  # Neutral chop - no clear regime
                long_cond = False
                short_cond = False
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: opposite condition or chop regime change
            if chop_val > 61.8:  # In chop, exit on WR normalization
                if wr_val > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif chop_val < 38.2:  # In trend, exit on trend reversal
                if close[i] < sma50[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Neutral - exit on opposite signal
                if wr_val > 50:  # Exit long condition
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit: opposite condition or chop regime change
            if chop_val > 61.8:  # In chop, exit on WR normalization
                if wr_val < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif chop_val < 38.2:  # In trend, exit on trend reversal
                if close[i] > sma50[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Neutral - exit on opposite signal
                if wr_val < 50:  # Exit short condition
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals