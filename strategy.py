#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and 1d volume confirmation.
# Long when price breaks above upper Donchian channel AND 1w EMA50 > EMA200 (bullish trend) AND 1d volume > 1.5 * 20-period average volume.
# Short when price breaks below lower Donchian channel AND 1w EMA50 < EMA200 (bearish trend) AND 1d volume > 1.5 * 20-period average volume.
# Exit when price retraces to the midpoint of the Donchian channel.
# Uses discrete position sizing (0.25) to limit fee churn. Designed for 1d timeframe with strict entry conditions.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d.

name = "1d_Donchian20_Breakout_1wEMA50_Trend_1dVolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 and EMA200 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_trend = align_htf_to_ltf(prices, df_1w, ema_50 > ema_200)  # Boolean: True for bullish, False for bearish
    
    # Calculate 1d volume confirmation filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Calculate Donchian channels (20-period)
    upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().values
    channel_mid = (upper_channel + lower_channel) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_trend[i]) or 
            np.isnan(volume_confirm[i]) or
            np.isnan(upper_channel[i]) or
            np.isnan(lower_channel[i]) or
            np.isnan(channel_mid[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above upper Donchian AND 1w EMA50 > EMA200 AND volume confirmation
            if (open_[i] <= upper_channel[i] and close[i] > upper_channel[i] and 
                ema_trend[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below lower Donchian AND 1w EMA50 < EMA200 AND volume confirmation
            elif (open_[i] >= lower_channel[i] and close[i] < lower_channel[i] and 
                  not ema_trend[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to Donchian channel midpoint
            if close[i] <= channel_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price retraces to Donchian channel midpoint
            if close[i] >= channel_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals