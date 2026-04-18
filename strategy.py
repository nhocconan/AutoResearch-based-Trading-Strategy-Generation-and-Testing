#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w EMA(20) trend filter + 20-bar high/low breakout with volume confirmation.
# Long when price breaks above 20-bar high with volume > 1.5x 20-bar average and price > 1w EMA(20).
# Short when price breaks below 20-bar low with volume > 1.5x 20-bar average and price < 1w EMA(20).
# Exit when price crosses back over 20-bar moving average.
# Uses weekly EMA for trend filter to avoid counter-trend trades and volume to confirm conviction.
# Designed for ~10-25 trades/year on daily timeframe.
name = "1d_1wEMA20_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # EMA(20) on 1w close
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 20-period high/low for breakout levels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period moving average for exit
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(ma_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_20_1w_aligned[i]
        high_level = high_20[i]
        low_level = low_20[i]
        ma_val = ma_20[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above 20-bar high with volume and trend alignment
            if close_val > high_level and vol_filter and close_val > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-bar low with volume and trend alignment
            elif close_val < low_level and vol_filter and close_val < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below 20-bar MA
            if close_val < ma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above 20-bar MA
            if close_val > ma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals