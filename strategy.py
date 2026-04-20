#!/usr/bin/env python3
"""
1h_Momentum_Filter_With_HTF_Trend_Session
Hypothesis: Use 1h RSI and momentum for entry timing, with 4h EMA trend filter and 1d volume regime filter.
Long when 1h RSI crosses above 50 with bullish momentum and 4h uptrend (price>EMA) and 1d volume above average.
Short when 1h RSI crosses below 50 with bearish momentum and 4h downtrend (price<EMA) and 1d volume above average.
Session filter: 08-20 UTC to avoid low-volume Asian session.
Target: 60-150 total trades over 4 years (15-37/year) with position size 0.20.
Works in bull/bear: 4h trend filter avoids counter-trend trades, volume filter ensures participation, session filter reduces noise.
"""

name = "1h_Momentum_Filter_With_HTF_Trend_Session"
timeframe = "1h"
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
    
    # Pre-calculate session hours (08-20 UTC)
    hours = prices.index.hour
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA34 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema34_4h = ema(close_4h, 34)
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume moving average (20-period)
    vol_ma20_1d = np.full_like(volume_1d, np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma20_1d[i] = np.mean(volume_1d[i-20:i])
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Calculate 1h RSI (14-period)
    def rsi(close_prices, period):
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close_prices, np.nan)
        avg_loss = np.full_like(close_prices, np.nan)
        
        if len(close_prices) >= period + 1:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            for i in range(period + 1, len(close_prices)):
                avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_values = 100 - (100 / (1 + rs))
        return rsi_values
    
    rsi_1h = rsi(close, 14)
    
    # Calculate 1h price momentum (rate of change over 3 periods)
    momentum = np.full_like(close, np.nan)
    for i in range(3, len(close)):
        momentum[i] = (close[i] - close[i-3]) / close[i-3] * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]) or 
            np.isnan(rsi_1h[i]) or np.isnan(momentum[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Volume filter: 1d volume above 20-period average
        volume_filter = volume[i] > vol_ma20_1d_aligned[i]
        
        if position == 0:
            # Long: RSI crosses above 50 with positive momentum, 4h uptrend, volume filter, session
            if (rsi_1h[i] > 50 and rsi_1h[i-1] <= 50 and 
                momentum[i] > 0 and 
                close[i] > ema34_4h_aligned[i] and 
                volume_filter and in_session):
                signals[i] = 0.20
                position = 1
            # Short: RSI crosses below 50 with negative momentum, 4h downtrend, volume filter, session
            elif (rsi_1h[i] < 50 and rsi_1h[i-1] >= 50 and 
                  momentum[i] < 0 and 
                  close[i] < ema34_4h_aligned[i] and 
                  volume_filter and in_session):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI drops below 40 OR 4h trend turns down
            if rsi_1h[i] < 40 or close[i] < ema34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI rises above 60 OR 4h trend turns up
            if rsi_1h[i] > 60 or close[i] > ema34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals