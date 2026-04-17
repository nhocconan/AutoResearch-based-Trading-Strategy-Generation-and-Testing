#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversal with 1d EMA200 trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) AND price > 1d EMA200 (uptrend) AND volume > 1.5x 20-period average.
Short when Williams %R > -20 (overbought) AND price < 1d EMA200 (downtrend) AND volume > 1.5x 20-period average.
Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts).
Williams %R identifies exhaustion points in both bull and bear markets, 1d EMA200 filters for primary trend,
volume confirmation reduces false signals. Designed for low trade frequency (12-30/year) to minimize fee drag
while capturing high-probability reversals at extremes.
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
    
    # Get 6h data for Williams %R calculation (14-period)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R on 6h timeframe: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_6h = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_6h = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_6h - close_6h) / (highest_high_6h - lowest_low_6h) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_6h - lowest_low_6h) == 0, -50, williams_r)
    
    # Calculate EMA200 on 1d timeframe
    close_1d_series = pd.Series(close_1d)
    ema_200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate volume average (20-period) on 6h
    volume_6h = df_6h['volume'].values
    volume_6h_series = pd.Series(volume_6h)
    volume_ma_6h = volume_6h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        ema_200 = ema_200_1d_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND price > 1d EMA200 (uptrend) AND volume > 1.5x avg
            if wr < -80 and price > ema_200 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < 1d EMA200 (downtrend) AND volume > 1.5x avg
            elif wr > -20 and price < ema_200 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses back above -50 (recovering from oversold)
            if wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses back below -50 (recovering from overbought)
            if wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA200_Volume_Filter"
timeframe = "6h"
leverage = 1.0