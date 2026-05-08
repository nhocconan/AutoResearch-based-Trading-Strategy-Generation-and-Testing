#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Choppiness Index regime filter with weekly trend direction
# Uses Choppiness Index (14) to detect ranging (CHOP > 61.8) vs trending (CHOP < 38.2) markets
# In ranging markets: mean reversion at Bollinger Bands (20,2)
# In trending markets: follow weekly trend direction using EMA(34) on weekly data
# Volume confirmation required for all entries to avoid false signals
# Designed to work in both bull and bear markets by adapting to regime
# Target: 50-150 total trades over 4 years = 12-37/year

name = "6h_ChopRegime_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend direction
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Choppiness Index (14) on 6h data
    def true_range(h, l, pc):
        tr1 = h - l
        tr2 = np.abs(h - pc)
        tr3 = np.abs(l - pc)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate true range
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = true_range(high, low, prev_close)
    
    # ATR(14)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    range_14 = highest_high - lowest_low
    # Avoid division by zero
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    chop = 100 * np.log10(sum_tr14 / range_14) / np.log10(14)
    
    # Bollinger Bands (20,2) for mean reversion in ranging markets
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(sma20[i]) or np.isnan(std20[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop[i]
        ema34_1w_val = ema34_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Ranging market: CHOP > 61.8 -> mean reversion at Bollinger Bands
            if chop_val > 61.8:
                if close[i] <= lower_bb[i] and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= upper_bb[i] and vol_spike:
                    signals[i] = -0.25
                    position = -1
            # Trending market: CHOP < 38.2 -> follow weekly trend
            elif chop_val < 38.2:
                if close[i] > ema34_1w_val and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < ema34_1w_val and vol_spike:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: opposite signal or regime change to extreme ranging
            if (chop_val > 61.8 and close[i] >= sma20[i]) or \
               (chop_val < 38.2 and close[i] < ema34_1w_val) or \
               (chop_val > 61.8 and close[i] >= upper_bb[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: opposite signal or regime change to extreme ranging
            if (chop_val > 61.8 and close[i] <= sma20[i]) or \
               (chop_val < 38.2 and close[i] > ema34_1w_val) or \
               (chop_val > 61.8 and close[i] <= lower_bb[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals