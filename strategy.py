#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams Alligator with 1d/1w regime filter + volume confirmation
    # Long: Alligator bullish alignment (jaw < teeth < lips) + price > 1d EMA200 + volume > 1.5x 20-period average
    # Short: Alligator bearish alignment (jaw > teeth > lips) + price < 1d EMA200 + volume > 1.5x 20-period average
    # Exit: Alligator convergence (jaws cross teeth) OR price crosses 1d EMA200
    # Using 12h timeframe for lower trade frequency, Williams Alligator for trend identification,
    # 1d EMA200 for strong trend filter, and volume confirmation to avoid false signals.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 with min_periods
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema_1d[199] = np.mean(close_1d[:200])  # SMA200 as seed
        multiplier = 2 / (200 + 1)
        for i in range(200, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Get 1w data for regime filter (ADX-like using EMA crossover)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 and EMA200 for regime filter
    ema_50_1w = np.full(len(close_1w), np.nan)
    ema_200_1w = np.full(len(close_1w), np.nan)
    
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        multiplier_50 = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * multiplier_50) + (ema_50_1w[i-1] * (1 - multiplier_50))
    
    if len(close_1w) >= 200:
        ema_200_1w[199] = np.mean(close_1w[:200])
        multiplier_200 = 2 / (200 + 1)
        for i in range(200, len(close_1w)):
            ema_200_1w[i] = (close_1w[i] * multiplier_200) + (ema_200_1w[i-1] * (1 - multiplier_200))
    
    # Calculate Williams Alligator on 12h data (periods: jaw=13, teeth=8, lips=5)
    # Jaw (13-period SMMA of median price)
    median_price = (high + low) / 2
    jaw = np.full(n, np.nan)
    if n >= 13:
        # Calculate SMMA (smoothed moving average) - similar to EMA but with different multiplier
        jaw[12] = np.mean(median_price[:13])
        multiplier_jaw = 1 / 13  # SMMA uses 1/period as multiplier
        for i in range(13, n):
            jaw[i] = (median_price[i] + jaw[i-1] * (13 - 1)) / 13
    
    # Teeth (8-period SMMA of median price)
    teeth = np.full(n, np.nan)
    if n >= 8:
        teeth[7] = np.mean(median_price[:8])
        multiplier_teeth = 1 / 8
        for i in range(8, n):
            teeth[i] = (median_price[i] + teeth[i-1] * (8 - 1)) / 8
    
    # Lips (5-period SMMA of median price)
    lips = np.full(n, np.nan)
    if n >= 5:
        lips[4] = np.mean(median_price[:5])
        multiplier_lips = 1 / 5
        for i in range(5, n):
            lips[i] = (median_price[i] + lips[i-1] * (5 - 1)) / 5
    
    # Align 1d EMA200 to 12h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Align 1w EMAs to 12h
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment conditions
        bullish_alligator = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])
        bearish_alligator = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        
        # Trend filter from 1d EMA200
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]
        
        # Regime filter from 1w EMA crossover (bullish when EMA50 > EMA200)
        bullish_regime = ema_50_1w_aligned[i] > ema_200_1w_aligned[i]
        bearish_regime = ema_50_1w_aligned[i] < ema_200_1w_aligned[i]
        
        # Entry logic: Alligator alignment + trend alignment + regime filter + volume confirmation
        long_entry = bullish_alligator and bullish_trend and bullish_regime and volume_spike[i]
        short_entry = bearish_alligator and bearish_trend and bearish_regime and volume_spike[i]
        
        # Exit logic: Alligator convergence (jaws cross teeth) OR trend reversal
        alligator_convergence = abs(jaw[i] - teeth[i]) < (jaw[i] * 0.001)  # 0.1% threshold
        long_exit = bearish_alligator or alligator_convergence or (close[i] < ema_1d_aligned[i])
        short_exit = bullish_alligator or alligator_convergence or (close[i] > ema_1d_aligned[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_1w_williams_alligator_ema200_volume_v1"
timeframe = "12h"
leverage = 1.0