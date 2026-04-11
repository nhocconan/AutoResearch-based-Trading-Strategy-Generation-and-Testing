#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d/1w trend filter with volume confirmation
# Long when price > Alligator teeth (green line) + 1d trend up + volume > 1.5x avg
# Short when price < Alligator teeth + 1d trend down + volume > 1.5x avg
# Exit when price crosses back below/above teeth or trend reverses
# Williams Alligator uses SMAs of median price: Jaw(13,8), Teeth(8,5), Lips(5,3)
# Designed for 15-35 trades/year on 12h timeframe with trend-following in both bull/bear markets

name = "12h_1w_alligator_trend_volume_v1"
timeframe = "12h"
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
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator on median price
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3)
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: both 1d and 1w EMAs must agree
        is_uptrend = close[i] > ema_50_1d_aligned[i] and close[i] > ema_50_1w_aligned[i]
        is_downtrend = close[i] < ema_50_1d_aligned[i] and close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions: price above/below teeth with lips alignment
        price_above_teeth = close[i] > teeth[i] and close[i] > lips[i]
        price_below_teeth = close[i] < teeth[i] and close[i] < lips[i]
        
        long_entry = price_above_teeth and volume_filter and is_uptrend
        short_entry = price_below_teeth and volume_filter and is_downtrend
        
        # Exit conditions: price crosses lips or trend changes
        long_exit = close[i] < lips[i] or not is_uptrend
        short_exit = close[i] > lips[i] or not is_downtrend
        
        # Priority: entry > exit > hold
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
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals