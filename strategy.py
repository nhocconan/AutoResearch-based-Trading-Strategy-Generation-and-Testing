#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (HMA21) and volume confirmation (>1.5x avg)
# - Long when price breaks above 4h 20-period high with volume > 1.5x 20-period average AND 1d HMA21 rising
# - Short when price breaks below 4h 20-period low with volume > 1.5x 20-period average AND 1d HMA21 falling
# - Exit when price retests 4h 10-period midpoint OR volume drops below average
# - 1d HMA21 trend filter ensures alignment with daily trend, reducing counter-trend trades
# - Volume confirmation prevents false breakouts in low-momentum environments
# - Targets 20-40 trades/year (80-160 total over 4 years) to stay within fee drag limits
# - Donchian breakouts capture momentum; HMA21 filter adds trend alignment; volume adds conviction

name = "4h_1d_donchian_breakout_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period) on 4h data
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    # 10-period midpoint for exit
    high_10 = prices['high'].rolling(window=10, min_periods=10).max().values
    low_10 = prices['low'].rolling(window=10, min_periods=10).min().values
    midpoint_10 = (high_10 + low_10) / 2.0
    
    # Pre-compute 1d HMA(21) for trend filter
    close_1d = df_1d['close'].values
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_1d).ewm(span=half_len, adjust=False, min_periods=half_len).mean()
    wma_full = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean()
    raw_hma = 2 * wma_half - wma_full
    hma_21 = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    # HMA slope: rising if current > previous, falling if current < previous
    hma_slope = np.zeros_like(hma_21_aligned)
    hma_slope[1:] = np.where(hma_21_aligned[1:] > hma_21_aligned[:-1], 1, -1)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < average volume for exit
    vol_normal = prices['volume'] < volume_20_avg
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(midpoint_10[i]) or np.isnan(hma_slope[i]) or
            np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > 20-day high with volume spike AND 1d HMA21 rising
            if (prices['close'].iloc[i] > high_20[i] and 
                vol_spike.iloc[i] and 
                hma_slope[i] > 0):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price < 20-day low with volume spike AND 1d HMA21 falling
            elif (prices['close'].iloc[i] < low_20[i] and 
                  vol_spike.iloc[i] and 
                  hma_slope[i] < 0):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retests 10-day midpoint (mean reversion signal)
            # 2. Volume drops below average (loss of momentum)
            if position == 1:  # Long position
                if (prices['close'].iloc[i] < midpoint_10[i] or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (prices['close'].iloc[i] > midpoint_10[i] or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals