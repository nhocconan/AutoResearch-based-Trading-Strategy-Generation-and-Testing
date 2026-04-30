#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Uses Donchian channel breakout as structure, 12h EMA50 for higher timeframe trend direction,
# and volume spike (>1.8x 20-bar avg) to filter false breakouts.
# Session filter (08-20 UTC) to trade only during liquid hours.
# Discrete position sizing at ±0.25 to manage fee drag.
# Target: 75-150 total trades over 4 years (19-38/year) to avoid excessive fees on 4h timeframe.
# Works in bull markets via breakout continuation and in bear markets via volatility expansion capture.

name = "4h_Donchian20_12hEMA50_Trend_VolumeConfirm_Session_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h_vals = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h_vals).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian Channel (20) on 4h
    donchian_period = 20
    upper_channel = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for Donchian and EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout above upper channel, price > 12h EMA50, volume spike
            if (curr_high > upper_channel[i] and 
                curr_close > curr_ema_50_12h and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout below lower channel, price < 12h EMA50, volume spike
            elif (curr_low < lower_channel[i] and 
                  curr_close < curr_ema_50_12h and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price returns to middle of Donchian channel (mean reversion)
            middle_channel = (upper_channel[i] + lower_channel[i]) / 2
            if curr_close < middle_channel:  # Price back below middle
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price returns to middle of Donchian channel
            middle_channel = (upper_channel[i] + lower_channel[i]) / 2
            if curr_close > middle_channel:  # Price back above middle
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals