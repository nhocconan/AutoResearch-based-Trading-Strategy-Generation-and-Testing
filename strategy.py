#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 12h EMA trend filter and volume confirmation.
# Long when Williams %R crosses above -50, price above 12h EMA(20), and volume > 1.5x 20-period average.
# Short when Williams %R crosses below -50, price below 12h EMA(20), and volume > 1.5x 20-period average.
# Exit when Williams %R crosses back below -50 (long) or above -50 (short).
# Williams %R identifies overbought/oversold conditions; EMA filter ensures trend alignment.
# Volume surge adds conviction. Designed for ~20-30 trades/year per symbol.
name = "4h_WilliamsR_12hEMA20_VolumeFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # EMA(20) on 12h close
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Williams %R (14-period) on 4h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_20_12h_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_20_12h_aligned[i]
        wr = williams_r[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: Williams %R crosses above -50, price above EMA, volume surge
            if i > start_idx and williams_r[i-1] <= -50 and wr > -50 and close_val > ema_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -50, price below EMA, volume surge
            elif i > start_idx and williams_r[i-1] >= -50 and wr < -50 and close_val < ema_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses back below -50
            if williams_r[i-1] > -50 and wr <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses back above -50
            if williams_r[i-1] < -50 and wr >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals