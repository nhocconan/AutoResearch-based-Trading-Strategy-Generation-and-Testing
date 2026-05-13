#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above 20-day high with 1w EMA50 trending up and volume > 1.5x average.
# Short when price breaks below 20-day low with 1w EMA50 trending down and volume > 1.5x average.
# Uses discrete sizing 0.25. Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe.
# Donchian channels provide clear structure. 1w EMA50 ensures we only trade with the weekly trend.
# Volume confirmation reduces false breakouts. Works in bull markets via upward breaks and in bear markets via downward breaks.

name = "1d_Donchian20_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    if n < lookback + 1:
        return np.zeros(n)
    
    # Donchian high and low (using previous period's data to avoid look-ahead)
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w data
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe (wait for 1w bar to close)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback + 20, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high with 1w EMA50 trending up and volume confirmation
            if (close[i] > donchian_high[i] and 
                close[i] > ema_50_1w_aligned[i] and  # Price above weekly EMA50 (uptrend)
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low with 1w EMA50 trending down and volume confirmation
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_50_1w_aligned[i] and  # Price below weekly EMA50 (downtrend)
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low (reversal) OR price crosses below weekly EMA50 (trend change)
            if (close[i] < donchian_low[i]) or (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high (reversal) OR price crosses above weekly EMA50 (trend change)
            if (close[i] > donchian_high[i]) or (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals