#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using Weekly Bollinger Band squeeze breakout with weekly trend filter.
# Long when price breaks above upper BB(20,2) on daily chart with weekly EMA50 uptrend and volume > 1.5x average.
# Short when price breaks below lower BB(20,2) with weekly EMA50 downtrend and volume > 1.5x average.
# Exit when price crosses the middle BB(20).
# Uses Bollinger Bands for volatility breakout, weekly EMA50 for trend filter, volume for confirmation.
# Target: 7-25 trades/year to avoid fee drag. Works in bull/bear via trend-aligned breakouts.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA50 trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Calculate Bollinger Bands (20, 2) on daily data
    bb_period = 20
    bb_std = 2
    
    # Middle band = SMA(20)
    sma = np.full(n, np.nan)
    for i in range(bb_period - 1, n):
        sma[i] = np.mean(close[i - bb_period + 1:i + 1])
    
    # Standard deviation
    bb_std_dev = np.full(n, np.nan)
    for i in range(bb_period - 1, n):
        bb_std_dev[i] = np.std(close[i - bb_period + 1:i + 1])
    
    upper_band = sma + bb_std * bb_std_dev
    lower_band = sma - bb_std * bb_std_dev
    
    # Calculate weekly EMA50 for trend filter
    ema_period = 50
    ema_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= ema_period:
        ema_weekly[ema_period - 1] = np.mean(close_weekly[:ema_period])
        for i in range(ema_period, len(close_weekly)):
            ema_weekly[i] = (close_weekly[i] * (2 / (ema_period + 1)) + 
                             ema_weekly[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Get volume MA for confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    # Align weekly indicators to daily timeframe
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need BB(20), weekly EMA50, and volume MA20
    start_idx = max(bb_period - 1, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(sma[i]) or np.isnan(ema_weekly_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: break above upper BB with weekly EMA50 uptrend and volume
            if (price > upper_band[i] and 
                price > ema_weekly_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: break below lower BB with weekly EMA50 downtrend and volume
            elif (price < lower_band[i] and 
                  price < ema_weekly_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below middle BB
            if price < sma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above middle BB
            if price > sma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyBollingerSqueeze_Breakout_WeeklyEMA50_Volume"
timeframe = "1d"
leverage = 1.0