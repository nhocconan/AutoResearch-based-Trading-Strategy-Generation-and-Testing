#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index regime + weekly trend filter with volume confirmation
# Uses Choppiness Index to detect ranging (CHOP > 61.8) vs trending (CHOP < 38.2) markets.
# In ranging markets: mean reversion at Bollinger Bands (20,2) with volume confirmation.
# In trending markets: follow weekly EMA(34) direction with volume confirmation.
# Designed to reduce whipsaws in sideways markets and capture trends with proper filters.
# Target: 20-80 total trades over 4 years = 5-20/year

name = "1d_ChopRegime_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def choppiness_index(high, low, close, period=14):
    """Choppiness Index: measures market choppiness vs trendiness"""
    atr = []
    for i in range(len(high)):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr.append(tr)
    
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum()
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    return chop.values

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
    
    # Calculate daily indicators
    chop = choppiness_index(high, low, close, 14)
    
    # Bollinger Bands (20,2)
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(bb_middle[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop[i]
        ema34_1w_val = ema34_1w_aligned[i]
        bb_upper_val = bb_upper[i]
        bb_lower_val = bb_lower[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Ranging market: CHOP > 61.8 -> mean reversion at BB with volume spike
            if chop_val > 61.8:
                if close[i] <= bb_lower[i] and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= bb_upper[i] and vol_spike:
                    signals[i] = -0.25
                    position = -1
            # Trending market: CHOP < 38.2 -> follow weekly trend with volume spike
            elif chop_val < 38.2:
                if close[i] > ema34_1w_val and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < ema34_1w_val and vol_spike:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: chop regime change or mean reversion signal
            if chop_val > 61.8 and close[i] >= bb_middle[i]:
                signals[i] = 0.0
                position = 0
            elif chop_val < 38.2 and close[i] < ema34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: chop regime change or mean reversion signal
            if chop_val > 61.8 and close[i] <= bb_middle[i]:
                signals[i] = 0.0
                position = 0
            elif chop_val < 38.2 and close[i] > ema34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals