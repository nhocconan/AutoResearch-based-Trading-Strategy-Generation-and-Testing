#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d EMA200 trend filter and volume confirmation.
# Long when %R < -80 (oversold) and price > 1d EMA200 with volume > 1.5x average.
# Short when %R > -20 (overbought) and price < 1d EMA200 with volume > 1.5x average.
# Exit when %R crosses back above -50 (for longs) or below -50 (for shorts).
# Williams %R identifies overextended moves likely to revert. EMA200 filter ensures
# alignment with long-term trend to avoid counter-trend trades in strong moves.
# Volume confirmation reduces false signals. Target: 20-40 trades/year for low frequency.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period) on 4h
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    # Avoid division by zero
    rr = highest_high - lowest_low
    willr = np.where(rr != 0, -100 * (highest_high - close) / rr, -50.0)
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1-day EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = max(200, 20)  # Ensure EMA200 and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(willr[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: %R oversold (< -80), above EMA200, volume confirmation
        if (willr[i] < -80 and 
            close[i] > ema200_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: %R overbought (> -20), below EMA200, volume confirmation
        elif (willr[i] > -20 and 
              close[i] < ema200_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: %R crosses back above -50 (for longs) or below -50 (for shorts)
        elif position == 1 and willr[i] > -50:
            signals[i] = 0.0
            position = 0
        elif position == -1 and willr[i] < -50:
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

name = "4h_WilliamsR_MeanRev_1dEMA200_VolumeFilter"
timeframe = "4h"
leverage = 1.0