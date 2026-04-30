#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high, price > 1d EMA50, and volume > 2x 20-bar avg.
# Short when price breaks below Donchian(20) low, price < 1d EMA50, and volume > 2x 20-bar avg.
# Exit when price crosses Donchian midpoint or on ATR-based stoploss.
# Donchian channels provide clear structural breakouts, EMA50 filters for higher timeframe trend,
# volume confirmation reduces false breakouts. Works in both bull (catch breakouts) and bear (fade breakdowns).
# Timeframe: 4h as per experiment guidelines.

name = "4h_Donchian20_1dEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_donchian_high = highest_high[i]
        curr_donchian_low = lowest_low[i]
        curr_donchian_mid = donchian_mid[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above Donchian high, price > 1d EMA50, volume spike
            if (curr_high > curr_donchian_high and 
                curr_close > curr_ema_50_1d and 
                curr_volume_confirm):
                signals[i] = 0.30
                position = 1
            # Short: Break below Donchian low, price < 1d EMA50, volume spike
            elif (curr_low < curr_donchian_low and 
                  curr_close < curr_ema_50_1d and 
                  curr_volume_confirm):
                signals[i] = -0.30
                position = -1
        
        elif position == 1:  # Long position
            # Exit conditions: price crosses below Donchian midpoint OR stoploss hit
            if curr_close < curr_donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit conditions: price crosses above Donchian midpoint OR stoploss hit
            if curr_close > curr_donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals