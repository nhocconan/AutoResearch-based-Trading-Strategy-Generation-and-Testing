#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with daily EMA trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; daily EMA filters trend direction.
# Volume confirmation reduces false signals. Designed for 50-150 total trades over 4 years.
# Works in bull (mean reversion in uptrend) and bear (mean reversion in downtrend) markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA and Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period) on daily data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_1d) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    # Calculate EMA(50) on daily close
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h Williams %R for entry timing (more responsive)
    highest_high_6h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_6h = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r_6h = np.where(
        (highest_high_6h - lowest_low_6h) != 0,
        ((highest_high_6h - close) / (highest_high_6h - lowest_low_6h)) * -100,
        -50
    )
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # need daily EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(williams_r_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter: price above daily EMA50 (uptrend) or below (downtrend)
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) in uptrend with volume
            if (williams_r_6h[i] < -80 and 
                trend_up and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) in downtrend with volume
            elif (williams_r_6h[i] > -20 and 
                  trend_down and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: Williams %R returns to neutral (> -50) or overbought (> -20)
            if williams_r_6h[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns to neutral (< -50) or oversold (< -80)
            if williams_r_6h[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_DailyEMA50_VolumeFilter"
timeframe = "6h"
leverage = 1.0