#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d volume spike and 1w EMA50 trend filter
# In ranging/accumulation phases: price breaks above R3 or below S3 with volume > 2x 20-period EMA = continuation signal
# Trend filter: only take longs when price > 1w EMA50 (bull bias), shorts when price < 1w EMA50 (bear bias)
# Uses discrete sizing (0.25) to minimize fee churn. Targets 50-150 total trades over 4 years.
# BTC/ETH edge: Camarilla levels from 1d capture institutional reaction zones; volume confirms breakout validity;
# weekly EMA50 prevents counter-trend trades in strong trends.

name = "6h_Camarilla_R3S3_Breakout_1dVolumeSpike_1wEMA50_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (using prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels from prior day's OHLC
    # Camarilla: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # We use prior day's values to avoid look-ahead
    prior_close = df_1d['close'].shift(1).values
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla R3 and S3 levels
    rang = prior_high - prior_low
    r3 = prior_close + (rang * 1.1 / 4)
    s3 = prior_close - (rang * 1.1 / 4)
    
    # Align 1d Camarilla levels to 6h timeframe (already aligned to prior day's close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long breakout: price > R3 with volume confirmation and bullish trend (price > 1w EMA50)
            if (close[i] > r3_aligned[i] and 
                volume_confirm and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price < S3 with volume confirmation and bearish trend (price < 1w EMA50)
            elif (close[i] < s3_aligned[i] and 
                  volume_confirm and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Camarilla H-L range (between S3 and R3) OR volume drops
            if (close[i] < r3_aligned[i] and close[i] > s3_aligned[i]) or volume[i] < vol_ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Camarilla H-L range OR volume drops
            if (close[i] < r3_aligned[i] and close[i] > s3_aligned[i]) or volume[i] < vol_ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals