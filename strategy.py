#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R extreme reversal with 1d EMA200 trend filter and volume spike confirmation.
Long when Williams %R < -80 (oversold) AND price > 1d EMA200 (bullish trend) AND volume > 2.0x 20-period average.
Short when Williams %R > -20 (overbought) AND price < 1d EMA200 (bearish trend) AND volume > 2.0x 20-period average.
Exit when Williams %R crosses back above -50 (for long) or below -50 (for short).
Designed for low trade frequency (19-50/year) to minimize fee drag while capturing mean reversals in both bull and bear markets.
Uses proven Williams %R edge from DB top performers.
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
    
    # Calculate Williams %R on 4h timeframe (primary)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 1d data for EMA200 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on 1d timeframe
    close_1d_series = pd.Series(close_1d)
    ema_200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate volume average (20-period) on 4h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)  # Note: volume_ma is 4d data, aligned via 1d reference
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)  # Williams %R is 4d data, aligned via 1d reference
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(williams_r_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_200 = ema_200_1d_aligned[i]
        vol_ma = volume_ma_aligned[i]
        wr = williams_r_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND price > 1d EMA200 (bullish trend) AND volume > 2.0x avg
            if wr < -80 and price > ema_200 and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < 1d EMA200 (bearish trend) AND volume > 2.0x avg
            elif wr > -20 and price < ema_200 and vol > 2.0 * vol_ma:
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

name = "4h_WilliamsR_Extreme_1dEMA200_Volume_Filter"
timeframe = "4h"
leverage = 1.0