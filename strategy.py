#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w trend filter and volume confirmation
# Long when price breaks above 6h Donchian high + 1w EMA21 uptrend + volume spike
# Short when price breaks below 6h Donchian low + 1w EMA21 downtrend + volume spike
# Uses discrete position sizing (0.25) to minimize fee churn
# Designed to work in both bull (trend continuation) and bear (mean reversion at extremes) markets

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    
    # Calculate weekly EMA(21) for trend filter
    weekly_close_series = pd.Series(weekly_close)
    ema_21_1w = weekly_close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align HTF indicators to 6h timeframe with proper delay
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate 6h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # Long: 6h breakout above Donchian high + 1w uptrend + volume confirmation
        # Short: 6h breakdown below Donchian low + 1w downtrend + volume confirmation
        # Discrete position sizing: 0.25
        
        # Long conditions
        if (close[i] > highest_20[i] and           # 6h breakout above Donchian high
            close[i] > ema_21_1w_aligned[i] and    # 1w uptrend (price above weekly EMA21)
            volume_ratio[i] > 1.5):                # Volume confirmation (1.5x average)
            signals[i] = 0.25
            
        # Short conditions
        elif (close[i] < lowest_20[i] and          # 6h breakdown below Donchian low
              close[i] < ema_21_1w_aligned[i] and  # 1w downtrend (price below weekly EMA21)
              volume_ratio[i] > 1.5):              # Volume confirmation (1.5x average)
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian_WeeklyEMA21_Trend_Volume_Filter"
timeframe = "6h"
leverage = 1.0