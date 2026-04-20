#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Williams %R with weekly trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; weekly trend ensures alignment with higher timeframe momentum
# Volume confirmation filters false signals. Works in both bull/bear markets by fading extremes in trending markets.
# Target: 10-25 trades/year with disciplined entries to avoid fee drag.

name = "1d_1w_WilliamsR_TrendFilter_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # === Daily Williams %R (14-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate highest high and lowest low over 14 days
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = -100 * (highest_high - close) / np.where((highest_high - lowest_low) != 0, (highest_high - lowest_low), 1)
    
    # === Weekly EMA(34) for trend filter ===
    weekly_close = df_1w['close'].values
    weekly_ema34 = pd.Series(weekly_close).ewm(span=34, min_periods=34, adjust=False).mean().values
    weekly_ema34_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema34)
    
    # === Daily Volume Spike Filter ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after weekly EMA warmup
        wr = williams_r[i]
        weekly_ema = weekly_ema34_aligned[i]
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(wr) or np.isnan(weekly_ema) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + weekly uptrend (price > weekly EMA34) + volume confirmation
            if wr < -80 and close_val > weekly_ema and vol_ratio_val > 1.8:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + weekly downtrend (price < weekly EMA34) + volume confirmation
            elif wr > -20 and close_val < weekly_ema and vol_ratio_val > 1.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns above -50 (momentum fading) OR weekly trend breaks
            if wr > -50 or close_val < weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns below -50 (momentum fading) OR weekly trend breaks
            if wr < -50 or close_val > weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals