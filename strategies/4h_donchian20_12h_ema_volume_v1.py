#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation
# Uses 4-hour Donchian channel breakouts for trend-following entries:
# - Buy when price breaks above 4h Donchian upper band (20-period high)
# - Sell when price breaks below 4h Donchian lower band (20-period low)
# - 12h EMA50 filter ensures trades align with higher timeframe trend
# - Volume confirmation avoids false breakouts
# Designed for moderate frequency (target: 20-50 trades/year) to balance edge and fees
# Works in both bull/bear via trend filter: follow 12h trend direction only

name = "4h_donchian20_12h_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 4h Donchian channel (20-period)
    # Calculate on 4h data then align
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 20-period high/low for Donchian
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    upper_band = align_htf_to_ltf(prices, df_4h, high_20)
    lower_band = align_htf_to_ltf(prices, df_4h, low_20)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if required data not available
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter from 12h EMA
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit on reverse breakout or when trend changes
            if close[i] < lower_band[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on reverse breakout or when trend changes
            if close[i] > upper_band[i] or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Breakout entries with trend and volume confirmation
            # Buy on upside breakout in uptrend
            if close[i] > upper_band[i] and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Sell on downside breakout in downtrend
            elif close[i] < lower_band[i] and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals