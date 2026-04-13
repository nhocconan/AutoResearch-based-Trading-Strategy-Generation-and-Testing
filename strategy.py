#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Bands with weekly trend filter and volume confirmation.
# Uses Bollinger Bands (20,2) for mean reversion in ranging markets and weekly EMA(40) for trend filter.
# In uptrends (price > weekly EMA40): look for long at lower BB band.
# In downtrends (price < weekly EMA40): look for short at upper BB band.
# Volume confirmation reduces false signals. Designed for low frequency (1d) to minimize fee drag.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # EMA(40) for 1w trend filter
    ema40_1w = np.zeros(len(close_1w))
    ema_multiplier = 2 / (40 + 1)
    ema40_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema40_1w[i] = (close_1w[i] - ema40_1w[i-1]) * ema_multiplier + ema40_1w[i-1]
    
    # Align 1w EMA to 1d timeframe
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Bollinger Bands (20,2) on 1d timeframe
    bb_period = 20
    bb_std = 2
    sma = np.full(n, np.nan)
    std = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(bb_period - 1, n):
        sma[i] = np.mean(close[i-bb_period+1:i+1])
        std[i] = np.std(close[i-bb_period+1:i+1])
        upper[i] = sma[i] + bb_std * std[i]
        lower[i] = sma[i] - bb_std * std[i]
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(bb_period, n):
        # Skip if any required data is not ready
        if (np.isnan(sma[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema40_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema40_1w_aligned[i]
        lower_band = lower[i]
        upper_band = upper[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: price at or below lower BB band + above weekly EMA40 + volume confirmation
            if (price <= lower_band and 
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price at or above upper BB band + below weekly EMA40 + volume confirmation
            elif (price >= upper_band and 
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above SMA (mean reversion target) or trend changes
            if (price >= sma[i] or
                price <= ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below SMA (mean reversion target) or trend changes
            if (price <= sma[i] or
                price >= ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_BollingerBands_Trend_Volume"
timeframe = "1d"
leverage = 1.0