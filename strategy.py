#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Breakout with 1d Volume Spike and Trend Filter
# Takes long when price breaks above upper Bollinger Band (20,2) with 1d volume spike and 1d close > SMA50
# Takes short when price breaks below lower Bollinger Band with 1d volume spike and 1d close < SMA50
# Exits when price returns to middle Bollinger Band
# Designed to capture volatility expansions in both bull and bear markets with volume confirmation
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 6h Bollinger Bands (20,2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    middle_bb = sma_20
    
    # Calculate 1d trend filter: SMA50
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    middle_bb_aligned = align_htf_to_ltf(prices, df_1d, middle_bb)
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for Bollinger and SMA calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(middle_bb_aligned[i]) or np.isnan(sma_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d_current = vol_1d[i] if i < len(vol_1d) else vol_1d[-1]
        
        if position == 0:
            # Long setup: break above upper BB with volume spike and bullish trend
            if (price > upper_bb_aligned[i] and 
                vol_1d_current > 2.0 * vol_ma_20_1d_aligned[i] and  # Volume spike
                close_1d[i] > sma_50_1d_aligned[i]):              # Above 1d SMA50
                position = 1
                signals[i] = position_size
            # Short setup: break below lower BB with volume spike and bearish trend
            elif (price < lower_bb_aligned[i] and 
                  vol_1d_current > 2.0 * vol_ma_20_1d_aligned[i] and  # Volume spike
                  close_1d[i] < sma_50_1d_aligned[i]):              # Below 1d SMA50
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle BB
            if price < middle_bb_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle BB
            if price > middle_bb_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Bollinger_Breakout_1dVolume_Trend"
timeframe = "6h"
leverage = 1.0