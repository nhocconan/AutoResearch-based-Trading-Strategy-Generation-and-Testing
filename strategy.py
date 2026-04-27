#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index regime + weekly EMA200 trend filter + volume spike.
# Long when CHOP > 61.8 (ranging market) + price > weekly EMA200 + volume spike.
# Short when CHOP > 61.8 (ranging market) + price < weekly EMA200 + volume spike.
# Exit when CHOP < 38.2 (trending market) or opposite signal.
# Uses weekly timeframe for trend filter to capture long-term bias and reduce noise.
# Target: 20-50 trades/year (80-200 over 4 years) to minimize fee decay while capturing mean reversion in ranges.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200 for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate daily Choppiness Index (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low since no previous close
    tr[0] = high[0] - low[0]
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop_raw = 100 * np.log10(atr * 14 / (max_high - min_low)) / np.log10(14)
    chop = np.where((max_high - min_low) > 0, chop_raw, 50.0)  # default to middle when range=0
    
    # Volume filter: volume > 2x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: choppy market (CHOP > 61.8) + above weekly EMA200 + volume spike
        if (chop[i] > 61.8 and 
            close[i] > ema200_1w_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: choppy market (CHOP > 61.8) + below weekly EMA200 + volume spike
        elif (chop[i] > 61.8 and 
              close[i] < ema200_1w_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: trending market (CHOP < 38.2) or opposite signal
        elif chop[i] < 38.2:
            signals[i] = 0.0
            position = 0
        elif position == 1 and close[i] < ema200_1w_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > ema200_1w_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_ChoppinessIndex_WeeklyEMA200_Trend_VolumeFilter"
timeframe = "1d"
leverage = 1.0