#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d Bollinger Band squeeze + volume spike + 1w trend filter.
# Long: Price below BB lower (20,2) + volume > 2x avg volume (20) + price > 1w EMA50
# Short: Price above BB upper (20,2) + volume > 2x avg volume (20) + price < 1w EMA50
# Uses 1d for mean reversion setup, 1w for trend filter, 6h for execution.
# Bollinger squeeze identifies low volatility compression; volume spike breaksout.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Bollinger Bands and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Bollinger Bands (20,2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # Average volume (20-period) for volume spike detection
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d Bollinger Bands to 6h
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    # Align 1d average volume to 6h
    avg_volume_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Align 1w EMA50 to 6h
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(avg_volume_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume_aligned[i]
        upper = bb_upper_aligned[i]
        lower = bb_lower_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        
        # Volume spike: current volume > 2x average volume
        volume_spike = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: price below BB lower + volume spike + above weekly EMA50
            if (price < lower and 
                volume_spike and
                price > ema_trend):
                position = 1
                signals[i] = position_size
            # Short: price above BB upper + volume spike + below weekly EMA50
            elif (price > upper and 
                  volume_spike and
                  price < ema_trend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above SMA20 (mean reversion target)
            sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
            sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20_1d)
            if price > sma_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below SMA20
            sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
            sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20_1d)
            if price < sma_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_1w_Bollinger_Squeeze_Volume"
timeframe = "6h"
leverage = 1.0