#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 1d EMA50 trend + Volume Spike
- Williams %R(14) < -90 for long, > -10 for short (extreme oversold/overbought)
- Trend filter: price > 1d EMA50 for long bias, price < 1d EMA50 for short bias
- Volume confirmation: volume > 2.0x 20-period MA to avoid false reversals
- Fixed position size 0.25 to limit fee churn and manage drawdown
- Works in bull markets (buying extreme dips in uptrends) and bear markets (selling extreme rallies in downtrends)
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 6h data for Williams %R and volume confirmation (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume average (20-period) on 6h
    volume_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe (primary)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema50 = ema50_aligned[i]
        wr = williams_r_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for extreme reversals with volume confirmation and trend filter
            # Long: Williams %R < -90 (extreme oversold) + price > EMA50 (uptrend) + volume spike
            if wr < -90 and price > ema50 and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -10 (extreme overbought) + price < EMA50 (downtrend) + volume spike
            elif wr > -10 and price < ema50 and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when Williams %R rises above -50 (momentum fading) or trend breaks
            if wr > -50 or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when Williams %R falls below -50 (momentum fading) or trend breaks
            if wr < -50 or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0