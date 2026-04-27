#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band squeeze breakout with weekly EMA20 trend filter and volume confirmation.
# Long when price breaks above upper BB with weekly EMA20 uptrend and volume > 1.5x average.
# Short when price breaks below lower BB with weekly EMA20 downtrend and volume > 1.5x average.
# Exit when price returns to middle BB.
# Uses Bollinger Bands for volatility-based breakouts, targeting 20-40 trades per year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA20 for trend filter
    ema_period = 20
    ema_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= ema_period:
        ema_weekly[ema_period - 1] = np.mean(close_weekly[:ema_period])
        for i in range(ema_period, len(close_weekly)):
            ema_weekly[i] = (close_weekly[i] * (2 / (ema_period + 1)) + 
                            ema_weekly[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2.0
    bb_middle = np.full(n, np.nan)
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    
    for i in range(bb_period - 1, n):
        bb_middle[i] = np.mean(close[i - bb_period + 1:i + 1])
        bb_std_dev = np.std(close[i - bb_period + 1:i + 1])
        bb_upper[i] = bb_middle[i] + bb_std_dev * bb_std
        bb_lower[i] = bb_middle[i] - bb_std_dev * bb_std
    
    # Align weekly EMA to daily timeframe
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need BB, weekly EMA, and volume MA20
    start_idx = max(bb_period, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_middle[i]) or np.isnan(ema_weekly_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price breaks above upper BB with weekly EMA20 uptrend and volume filter
            if (price > bb_upper[i] and price > ema_weekly_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: price breaks below lower BB with weekly EMA20 downtrend and volume filter
            elif (price < bb_lower[i] and price < ema_weekly_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle BB
            if price <= bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to middle BB
            if price >= bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_BollingerBandSqueezeBreakout_WeeklyEMA20_Volume"
timeframe = "1d"
leverage = 1.0