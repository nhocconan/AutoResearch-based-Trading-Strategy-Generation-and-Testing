#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel with 1w uptrend (price > 1w EMA34) and volume spike (>2.0x 20-bar avg).
# Short when price breaks below lower Donchian channel with 1w downtrend (price < 1w EMA34) and volume spike.
# Exit when price returns to the middle of the Donchian channel (mean reversion).
# Uses proven Donchian structure from top performers, 1w EMA34 for HTF trend, and strict volume confirmation.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

name = "1d_Donchian20_1wEMA34_Trend_VolumeConfirmation_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian channels (20-period) on 1d data
    # Upper channel = highest high over past 20 periods
    # Lower channel = lowest low over past 20 periods
    # Middle channel = average of upper and lower
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    middle_channel = (upper_channel + lower_channel) / 2.0
    
    # Volume confirmation: volume > 2.0x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA34 and Donchian channels
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(middle_channel[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_upper = upper_channel[i]
        curr_lower = lower_channel[i]
        curr_middle = middle_channel[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian level, uptrend (price > 1w EMA34), volume spike
            if (curr_close > curr_upper and 
                curr_close > curr_ema_34_1w and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian level, downtrend (price < 1w EMA34), volume spike
            elif (curr_close < curr_lower and 
                  curr_close < curr_ema_34_1w and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price returns to middle of Donchian channel (mean reversion)
            if curr_close <= curr_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price returns to middle of Donchian channel (mean reversion)
            if curr_close >= curr_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals