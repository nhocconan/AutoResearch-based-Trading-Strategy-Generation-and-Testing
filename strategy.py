#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_elder_ray_power_v2"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
        return signals
    
    # Calculate Elder Ray components
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 13-period EMA on weekly data for trend filter
    close_1w = df_1w['close'].values
    ema13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull and Bear Power for daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align daily indicators to 6h timeframe
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Align weekly EMA to 6h timeframe
    ema13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema13_1w)
    
    # Volume confirmation: 6h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(ema13_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        ema13 = ema13_1d_aligned[i]
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        ema13_weekly = ema13_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * vol_ma_20[i]
        
        # Trend filter: price above/below weekly EMA13
        uptrend = price_close > ema13_weekly
        downtrend = price_close < ema13_weekly
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bull Power > 0 (bulls in control) + price above daily EMA13 + uptrend on weekly + volume
        if bull > 0 and price_close > ema13 and uptrend and vol_confirm:
            enter_long = True
        
        # Short: Bear Power < 0 (bears in control) + price below daily EMA13 + downtrend on weekly + volume
        if bear < 0 and price_close < ema13 and downtrend and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite power signal or trend change
        exit_long = bull < 0 or not uptrend  # Bears take over or trend turns down
        exit_short = bear > 0 or not downtrend  # Bulls take over or trend turns up
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 6h Elder Ray power strategy with weekly trend filter.
# Uses daily Bull/Bear Power (High/Low - EMA13) to measure bull/bear strength.
# Enters long when Bull Power > 0 (bulls in control) with price above daily EMA13,
# weekly uptrend, and volume confirmation. Enters short when Bear Power < 0
# (bears in control) with price below daily EMA13, weekly downtrend, and volume.
# Weekly EMA13 filter ensures we trade with the higher timeframe trend.
# Works in both bull and bear markets by adapting to the prevailing trend.
# Position size 0.25 limits drawdown during volatile periods.
# Target: 15-25 trades per year (60-100 total over 4 years) to minimize fee drag.