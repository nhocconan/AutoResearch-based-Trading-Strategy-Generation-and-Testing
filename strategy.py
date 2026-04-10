#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above R4 with volume > 1.8x average AND weekly close > weekly EMA34
# - Short when price breaks below S4 with volume > 1.8x average AND weekly close < weekly EMA34
# - Exit when price retests daily pivot point or volume drops below average
# - Weekly trend filter ensures alignment with major trend
# - Volume confirmation prevents false breakouts
# - Targets 12-30 trades/year (48-120 total over 4 years) to avoid fee drag
# - Camarilla pivots from 1d provide structure, 6h timeframe reduces noise, 1w filter catches major trends

name = "6h_1w_camarilla_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla pivot levels
    high_1d = prices['high'].rolling(window=1, min_periods=1).max()  # daily high
    low_1d = prices['low'].rolling(window=1, min_periods=1).min()    # daily low
    close_1d = prices['close'].rolling(window=1, min_periods=1).mean()  # daily close
    
    # Calculate daily pivot and ranges
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    R4 = close_1d + (range_1d * 1.1 / 2)
    R3 = close_1d + (range_1d * 1.1 / 4)
    S3 = close_1d - (range_1d * 1.1 / 4)
    S4 = close_1d - (range_1d * 1.1 / 2)
    
    # Align to 6h timeframe
    R4_6h = align_htf_to_ltf(prices, prices, R4.values)
    R3_6h = align_htf_to_ltf(prices, prices, R3.values)
    S3_6h = align_htf_to_ltf(prices, prices, S3.values)
    S4_6h = align_htf_to_ltf(prices, prices, S4.values)
    pivot_6h = align_htf_to_ltf(prices, prices, pivot_1d.values)
    
    # Pre-compute 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Pre-compute volume confirmation: > 1.8x 24-period average (4 days)
    volume_24_avg = prices['volume'].rolling(window=24, min_periods=24).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_24_avg)
    
    # Pre-compute volume filter: < average volume for exit
    vol_normal = prices['volume'] < volume_24_avg
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if any required data is invalid
        if (np.isnan(R4_6h[i]) or np.isnan(S4_6h[i]) or 
            np.isnan(pivot_6h[i]) or np.isnan(ema34_1w_aligned[i]) or
            np.isnan(volume_24_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > R4 with volume spike AND weekly uptrend
            if (prices['close'].iloc[i] > R4_6h[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema34_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price < S4 with volume spike AND weekly downtrend
            elif (prices['close'].iloc[i] < S4_6h[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema34_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retests daily pivot point (mean reversion signal)
            # 2. Volume drops below average (loss of momentum)
            if position == 1:  # Long position
                if (prices['close'].iloc[i] < pivot_6h[i] or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (prices['close'].iloc[i] > pivot_6h[i] or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals