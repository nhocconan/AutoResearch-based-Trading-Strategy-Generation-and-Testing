#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1-day trend filter
# Bull Power = High - EMA13; Bear Power = Low - EMA13
# Long when Bull Power > 0 and daily close > daily EMA50 (uptrend)
# Short when Bear Power < 0 and daily close < daily EMA50 (downtrend)
# Uses volume confirmation to avoid low-conviction moves
# Target: 50-150 trades over 4 years, works in bull/bear via trend filter

name = "6h_1d_elder_ray_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray on daily
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate EMA50 for trend filter on daily
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Bull Power = High - EMA13
    bull_power_1d = df_1d['high'].values - ema13_1d
    
    # Bear Power = Low - EMA13
    bear_power_1d = df_1d['low'].values - ema13_1d
    
    # Trend filter: daily close > EMA50 for uptrend, < EMA50 for downtrend
    uptrend_1d = close_1d > ema50_1d
    downtrend_1d = close_1d < ema50_1d
    
    # Align all indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Load 6h data for volume confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 50 to ensure sufficient data
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(uptrend_aligned[i]) or np.isnan(downtrend_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 20-period average
        vol_confirm = volume[i] > vol_avg_20[i]
        
        # Entry conditions
        long_signal = vol_confirm and (bull_power_aligned[i] > 0) and uptrend_aligned[i] > 0.5
        short_signal = vol_confirm and (bear_power_aligned[i] < 0) and downtrend_aligned[i] > 0.5
        
        # Exit conditions: power crosses zero or trend changes
        long_exit = (bull_power_aligned[i] <= 0) or (uptrend_aligned[i] <= 0.5)
        short_exit = (bear_power_aligned[i] >= 0) or (downtrend_aligned[i] <= 0.5)
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals