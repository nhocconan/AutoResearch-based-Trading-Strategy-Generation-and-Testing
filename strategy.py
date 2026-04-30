#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
# Long when price breaks above upper Donchian channel with 1d uptrend (price > 1d EMA50) and volume spike (>2.0x 20-bar avg).
# Short when price breaks below lower Donchian channel with 1d downtrend (price < 1d EMA50) and volume spike.
# Exit when price returns to the middle of the Donchian channel (mean reversion).
# Uses proven Donchian structure with strict volume confirmation to limit trades.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

name = "12h_Donchian20_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Previous 1d OHLC for volatility context (completed 1d bar)
    df_1d_prev = get_htf_data(prices, '1d')
    if len(df_1d_prev) < 2:
        return np.zeros(n)
    
    prev_high_1d = df_1d_prev['high'].shift(1).values
    prev_low_1d = df_1d_prev['low'].shift(1).values
    
    # Align 1d data to 12h timeframe (completed 1d bar only)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d_prev, prev_high_1d)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d_prev, prev_low_1d)
    
    # Calculate ATR-based volatility filter (use previous 1d range)
    prev_range = prev_high_aligned - prev_low_aligned
    volatility_filter = prev_range > (0.5 * pd.Series(prev_range).rolling(window=20, min_periods=20).mean().values)
    
    # Donchian(20) channels on 12h timeframe
    lookback = 20
    upper_channel = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower_channel = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    middle_channel = (upper_channel + lower_channel) / 2
    
    # Volume confirmation: volume > 2.0x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Donchian channels
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(middle_channel[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(volatility_filter[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_upper = upper_channel[i]
        curr_lower = lower_channel[i]
        curr_middle = middle_channel[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        curr_volatility_filter = volatility_filter[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper channel, uptrend (price > 1d EMA50), volume spike, volatility filter
            if (curr_close > curr_upper and 
                curr_close > curr_ema_50_1d and 
                curr_volume_confirm and 
                curr_volatility_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel, downtrend (price < 1d EMA50), volume spike, volatility filter
            elif (curr_close < curr_lower and 
                  curr_close < curr_ema_50_1d and 
                  curr_volume_confirm and 
                  curr_volatility_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price returns to middle of channel (mean reversion)
            if curr_close <= curr_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price returns to middle of channel (mean reversion)
            if curr_close >= curr_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals