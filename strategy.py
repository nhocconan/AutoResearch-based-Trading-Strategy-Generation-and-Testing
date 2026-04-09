#!/usr/bin/env python3
# 1h_4h1d_trend_with_volume_and_session_filter_v1
# Hypothesis: Use 4h EMA(21) for trend direction and 1d Donchian(20) for breakout structure, with 1h RSI(14) pullback entries and volume confirmation (>1.5x 20-period average). Trade only during 08-20 UTC session to reduce noise. Discrete position size 0.20. Target 60-150 trades over 4 years (15-37/year) by requiring 4h trend alignment, 1d breakout, RSI extreme, volume spike, and session filter. Works in bull (trend+breakout) and bear (mean reversion at extremes via RSI).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_trend_with_volume_and_session_filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 4h EMA(21) for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    ema_21_4h = pd.Series(df_4h['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1d Donchian(20) for breakout structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    highest_20_1d = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_20_1d = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    highest_20_1d_aligned = align_htf_to_ltf(prices, df_1d, highest_20_1d)
    lowest_20_1d_aligned = align_htf_to_ltf(prices, df_1d, lowest_20_1d)
    
    # 1h RSI(14) for pullback entries
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(ema_21_4h_aligned[i]) or
            np.isnan(highest_20_1d_aligned[i]) or np.isnan(lowest_20_1d_aligned[i]) or
            np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or price breaks below 4h EMA
            if rsi_values[i] > 70 or close[i] < ema_21_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or price breaks above 4h EMA
            if rsi_values[i] < 30 or close[i] > ema_21_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter only with volume confirmation and session
            if volume_confirmed:
                # Long: price breaks above 1d Donchian high AND 4h EMA uptrend AND RSI < 40 (pullback)
                if (close[i] > highest_20_1d_aligned[i] and 
                    ema_21_4h_aligned[i] > ema_21_4h_aligned[i-1] and  # EMA rising
                    rsi_values[i] < 40):
                    position = 1
                    signals[i] = 0.20
                # Short: price breaks below 1d Donchian low AND 4h EMA downtrend AND RSI > 60 (pullback)
                elif (close[i] < lowest_20_1d_aligned[i] and 
                      ema_21_4h_aligned[i] < ema_21_4h_aligned[i-1] and  # EMA falling
                      rsi_values[i] > 60):
                    position = -1
                    signals[i] = -0.20
    
    return signals