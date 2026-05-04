#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter (EMA34) and volume confirmation (>2.0x 20 EMA volume)
# Uses 1d Donchian channel breakouts for structure - captures strong momentum bursts
# 1w EMA34 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation filters false breakouts (>2.0x average volume) - tight to reduce trades
# Discrete sizing 0.25 minimizes fee churn while maintaining profitability
# Target: 30-100 total trades over 4 years = 7-25/year for 1d timeframe
# Works in bull markets (continuation at upper channel) and bear markets (continuation at lower channel)
# Focus on BTC/ETH by requiring 1w trend alignment (avoids SOL-only bias)

name = "1d_Donchian20_1wEMA34_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need enough data for EMA34 calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(34) trend filter from prior completed 1w bar
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_shifted = np.roll(ema_34_1w, 1)
    ema_34_1w_shifted[0] = np.nan
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d Donchian channels (20-period) from prior completed 1d bar
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Upper channel: 20-period high
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    # Lower channel: 20-period low
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only prior completed 1d bar (no look-ahead)
    upper_channel_shifted = np.roll(upper_channel, 1)
    lower_channel_shifted = np.roll(lower_channel, 1)
    upper_channel_shifted[0] = np.nan
    lower_channel_shifted[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(upper_channel_shifted[i]) or np.isnan(lower_channel_shifted[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper channel AND price > 1w EMA34 AND volume spike
            if close[i] > upper_channel_shifted[i] and close[i] > ema_34_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower channel AND price < 1w EMA34 AND volume spike
            elif close[i] < lower_channel_shifted[i] and close[i] < ema_34_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to lower channel OR price crosses below 1w EMA34
            if close[i] < lower_channel_shifted[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to upper channel OR price crosses above 1w EMA34
            if close[i] > upper_channel_shifted[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals