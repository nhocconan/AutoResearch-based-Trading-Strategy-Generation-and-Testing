#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R with 4h trend filter (EMA200) and volume confirmation.
# Williams %R identifies overbought/oversold conditions for mean reversion entries.
# 4h EMA200 provides trend bias to avoid counter-trend trades.
# Volume > 1.3x average confirms institutional participation.
# Works in bull/bear as 4h EMA200 adapts to trend.
# Target: 15-37 trades/year per symbol (60-150 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA(200) for trend filter
    ema_len = 200
    if len(df_4h) < ema_len:
        return np.zeros(n)
    
    ema_4h = pd.Series(df_4h['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Williams %R (14 periods) on 1h
    wr_len = 14
    highest_high = pd.Series(high).rolling(window=wr_len, min_periods=wr_len).max().values
    lowest_low = pd.Series(low).rolling(window=wr_len, min_periods=wr_len).min().values
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = max(50, wr_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(wr[i]) or 
            np.isnan(ema_4h_aligned[i]) or
            np.isnan(vol_ma[i]) or
            not session_mask[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 4h EMA200
        above_ema = close[i] > ema_4h_aligned[i]
        below_ema = close[i] < ema_4h_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: Williams %R oversold (< -80) + above 4h EMA200 + volume
            if (wr[i] < -80 and 
                above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Williams %R overbought (> -20) + below 4h EMA200 + volume
            elif (wr[i] > -20 and 
                  below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to -50 or breaks below 4h EMA200
            if wr[i] > -50 or close[i] < ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to -50 or breaks above 4h EMA200
            if wr[i] < -50 or close[i] > ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_WilliamsR_4hEMA200_Volume_v1"
timeframe = "1h"
leverage = 1.0