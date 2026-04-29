#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA200 trend filter + volume confirmation
# Long when price breaks above Donchian(20) high, close > 1w EMA200, and volume > 1.5x 20-period average
# Short when price breaks below Donchian(20) low, close < 1w EMA200, and volume > 1.5x 20-period average
# Exit when price crosses Donchian(10) midpoint (mean reversion) or trend filter fails
# Uses discrete position sizing (0.30) to balance capture and risk.
# Donchian channels provide clear breakout levels, 1w EMA200 filters for higher-timeframe trend,
# volume confirmation ensures breakout validity.
# Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years) to avoid overtrading.
# Works in both bull and bear markets by only trading breakouts in the direction of the 1w trend.

name = "1d_Donchian20_1wEMA200_Volume_v1"
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
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Donchian channels (20-period for entry, 10-period for exit)
    # Donchian high: highest high over past 20 periods
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low over past 20 periods
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint for exit: average of 10-period high and low
    donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    donchian_mid_10 = (donchian_high_10 + donchian_low_10) / 2.0
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 200)  # Donchian20 and 1w EMA200 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(volume_threshold[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_donchian_high_20 = donchian_high_20[i]
        curr_donchian_low_20 = donchian_low_20[i]
        curr_donchian_mid_10 = donchian_mid_10[i]
        curr_ema200_1w = ema_200_1w_aligned[i]
        curr_vol_threshold = volume_threshold[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below Donchian(10) midpoint OR trend filter fails (close < 1w EMA200)
            if curr_close < curr_donchian_mid_10 or curr_close < curr_ema200_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian(10) midpoint OR trend filter fails (close > 1w EMA200)
            if curr_close > curr_donchian_mid_10 or curr_close > curr_ema200_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian(20) high, close > 1w EMA200, and volume confirmation
            if (curr_high > curr_donchian_high_20 and 
                curr_close > curr_ema200_1w and 
                curr_vol > curr_vol_threshold):
                signals[i] = 0.30
                position = 1
            # Short when price breaks below Donchian(20) low, close < 1w EMA200, and volume confirmation
            elif (curr_low < curr_donchian_low_20 and 
                  curr_close < curr_ema200_1w and 
                  curr_vol > curr_vol_threshold):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals