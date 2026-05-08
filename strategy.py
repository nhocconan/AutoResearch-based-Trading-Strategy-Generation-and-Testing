#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Donchian_Trend_Volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_weekly = get_htf_data(prices, '1w')
    
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Weekly Donchian channels (20 periods)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate Donchian upper and lower bands
    upper_band = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe (waits for weekly close)
    upper_band_daily = align_htf_to_ltf(prices, df_weekly, upper_band)
    lower_band_daily = align_htf_to_ltf(prices, df_weekly, lower_band)
    
    # Weekly EMA20 for trend filter
    ema20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_weekly_daily = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # Daily volume filter: current volume > 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_band_daily[i]) or np.isnan(lower_band_daily[i]) or
            np.isnan(ema20_weekly_daily[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper band with trend filter and volume
            long_cond = (close[i] > upper_band_daily[i] and 
                        close[i] > ema20_weekly_daily[i] and
                        volume[i] > vol_ma20[i])
            
            # Short: break below lower band with trend filter and volume
            short_cond = (close[i] < lower_band_daily[i] and 
                         close[i] < ema20_weekly_daily[i] and
                         volume[i] > vol_ma20[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below weekly EMA20 or breakdown below lower band
            if close[i] < ema20_weekly_daily[i] or close[i] < lower_band_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above weekly EMA20 or breakout above upper band
            if close[i] > ema20_weekly_daily[i] or close[i] > upper_band_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Donchian breakout with trend filter and volume confirmation.
# In bull markets: captures strong uptrends via breakouts above weekly resistance.
# In bear markets: captures sharp declines via breakouts below weekly support.
# Weekly EMA20 ensures we only trade in direction of weekly trend.
# Volume confirmation reduces false breakouts.
# Target: 20-50 trades over 4 years (5-12/year) to minimize fee drag.
# Works on BTC/ETH via institutional weekly support/resistance levels.