#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly trend filter + daily price action + volume confirmation.
# Long when price closes above daily open AND weekly EMA(34) AND volume > 1.5x daily average.
# Short when price closes below daily open AND weekly EMA(34) AND volume > 1.5x daily average.
# Exit when price crosses back below/above daily open.
# Uses weekly EMA for trend filter, daily open-close for price action, volume for conviction.
# Designed for ~10-25 trades/year per symbol to avoid fee drag.
name = "1d_weeklyEMA34_dailyOpenClose_volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # EMA(34) on weekly close
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily average volume (20-day average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        open_val = open_price[i]
        ema_val = ema_34_1w_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: close above open AND above weekly EMA with volume surge
            if close_val > open_val and close_val > ema_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: close below open AND below weekly EMA with volume surge
            elif close_val < open_val and close_val < ema_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close back below open
            if close_val < open_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close back above open
            if close_val > open_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals