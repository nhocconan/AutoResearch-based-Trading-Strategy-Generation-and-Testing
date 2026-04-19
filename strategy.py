#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h breakout strategy using 4h Donchian channels with 1d trend filter and volume confirmation.
# Uses higher timeframes for signal direction (4h trend, 1d Donchian breakout) and 1h for entry timing.
# Volume confirmation reduces false breakouts. Designed for 15-35 trades/year to avoid fee drag.
# Works in bull/bear markets: trend filter ensures we only trade with higher timeframe momentum.
name = "1h_Donchian_Trend_Filter_Volume_Breakout"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (Donchian middle)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channel (20-period)
    donchian_period = 20
    upper_4h = pd.Series(high_4h).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_4h = pd.Series(low_4h).rolling(window=donchian_period, min_periods=donchian_period).min().values
    middle_4h = (upper_4h + lower_4h) / 2.0
    
    # Get 1d data for breakout direction filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Donchian breakout levels (15-period)
    donchian_1d_period = 15
    upper_1d = pd.Series(high_1d).rolling(window=donchian_1d_period, min_periods=donchian_1d_period).max().values
    lower_1d = pd.Series(low_1d).rolling(window=donchian_1d_period, min_periods=donchian_1d_period).min().values
    
    # Align HTF indicators to 1h
    middle_4h_aligned = align_htf_to_ltf(prices, df_4h, middle_4h)
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # Volume confirmation: current volume > 1.8x 24-period average (to reduce noise)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_period, donchian_1d_period, 24)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(middle_4h_aligned[i]) or np.isnan(upper_1d_aligned[i]) or 
            np.isnan(lower_1d_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_24[i]
        middle_4h_val = middle_4h_aligned[i]
        upper_1d_val = upper_1d_aligned[i]
        lower_1d_val = lower_1d_aligned[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long: price above 4h Donchian middle AND breaks above 1d upper band with volume
            if price > middle_4h_val and price > upper_1d_val and volume_confirmed:
                signals[i] = 0.20
                position = 1
            # Short: price below 4h Donchian middle AND breaks below 1d lower band with volume
            elif price < middle_4h_val and price < lower_1d_val and volume_confirmed:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long when price returns to 4h Donchian middle (mean reversion)
            if price < middle_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short when price returns to 4h Donchian middle
            if price > middle_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals