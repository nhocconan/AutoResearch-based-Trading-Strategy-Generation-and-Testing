#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above upper Donchian channel (20d) AND price > 1w EMA50 AND volume spike
# Short when price breaks below lower Donchian channel (20d) AND price < 1w EMA50 AND volume spike
# Donchian channels provide clear structure for breakouts in both bull and bear markets
# 1w EMA50 offers smooth long-term trend filter to reduce whipsaw and avoid counter-trend trades
# Volume spike requires 2.0x 20-bar MA for confirmation (balanced to avoid overtrading)
# Target: 50-100 total trades over 4 years (12-25/year) to minimize fee drag while capturing major moves
# Works in bull (trend + breakouts) and bear (mean reversion at extremes + volume confirmation)
# Timeframe: 1d (primary timeframe as required)

name = "1d_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels from 1d data (using previous completed daily bar)
    # We need 20 periods of high/low for the channels
    if len(close) < 20:
        return np.zeros(n)
    
    # Calculate rolling max/min for high/low over 20 periods
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed daily bar (look-ahead safety)
    high_roll_max_shifted = np.roll(high_roll_max, 1)
    low_roll_min_shifted = np.roll(low_roll_min, 1)
    
    # Upper channel = previous 20-period high, Lower channel = previous 20-period low
    upper_channel = high_roll_max_shifted
    lower_channel = low_roll_min_shifted
    
    # Volume confirmation on 1d (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)  # Volume spike threshold
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN (due to roll or insufficient data)
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper channel AND strong uptrend (price > 1w EMA50) AND volume spike
            if (close[i] > upper_channel[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel AND strong downtrend (price < 1w EMA50) AND volume spike
            elif (close[i] < lower_channel[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below upper channel OR closes below 1w EMA50
            if close[i] < upper_channel[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above lower channel OR closes above 1w EMA50
            if close[i] > lower_channel[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals