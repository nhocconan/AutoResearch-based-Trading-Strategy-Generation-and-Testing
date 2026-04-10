#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# - Long when price breaks above Donchian(20) high + weekly pivot bias bullish + 6h volume > 1.5x 20-period volume SMA
# - Short when price breaks below Donchian(20) low + weekly pivot bias bearish + 6h volume > 1.5x 20-period volume SMA
# - Exit: price crosses Donchian(20) midline (10-period average of high/low)
# - Position sizing: 0.25 discrete level
# - Works in bull/bear: Donchian captures breakouts in trending markets, weekly pivot filters counter-trend breaks
# - Volume confirmation ensures institutional participation, reducing false breakouts
# - Weekly pivot derived from prior week's OHLC provides structural bias

name = "6h_1w_donchian_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 10 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Weekly pivot bias: bullish if close > weekly_pivot, bearish if close < weekly_pivot
    weekly_bias = np.where(weekly_close > weekly_pivot, 1, np.where(weekly_close < weekly_pivot, -1, 0))
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias.astype(float))
    
    # Calculate 6h volume SMA(20) for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(weekly_bias_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Weekly pivot bias
        bias_bullish = weekly_bias_aligned[i] == 1
        bias_bearish = weekly_bias_aligned[i] == -1
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        # Exit condition: price crosses Donchian midline
        cross_mid_up = close[i] > donchian_mid[i] and close[i-1] <= donchian_mid[i-1]
        cross_mid_down = close[i] < donchian_mid[i] and close[i-1] >= donchian_mid[i-1]
        
        # Entry conditions
        long_entry = breakout_up and bias_bullish and vol_confirm
        short_entry = breakout_down and bias_bearish and vol_confirm
        
        # Exit conditions
        exit_long = cross_mid_down
        exit_short = cross_mid_up
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals