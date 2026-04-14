# 4h_Volume_Price_Action_Range_Breakout
# Strategy: Mean reversion within daily range using 1h structure confirmation
# Buy near 1d low with bullish 1h candle and volume spike
# Sell near 1d high with bearish 1h candle and volume spike
# Designed for range-bound markets (2025+) with controlled frequency
# Target: 25-40 trades/year to minimize fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1h EMA for short-term trend (21-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    ema_1h = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 1h ATR for volatility (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Get 1h data for structure analysis
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:
        return np.zeros(n)
    
    # Calculate 1h body size (close-open) for candle strength
    open_1h = df_1h['open'].values
    close_1h = df_1h['close'].values
    body_size = np.abs(close_1h - open_1h)
    body_mean = pd.Series(body_size).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Get daily data for range calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily range (high-low)
    high_1d = df_1h['high'].values  # Actually using 1h high/low for intraday range
    low_1d = df_1h['low'].values
    daily_range = high_1d - low_1d
    range_ma = pd.Series(daily_range).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation (20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(30, n):
        # Get aligned 1h indicators
        ema_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_1h)[i]
        body_mean_aligned = align_htf_to_ltf(prices, df_1h, body_mean)[i]
        range_ma_aligned = align_htf_to_ltf(prices, df_1h, range_ma)[i]
        
        if np.isnan(ema_1h_aligned) or np.isnan(body_mean_aligned) or np.isnan(range_ma_aligned) or \
           np.isnan(atr_1h[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Volume confirmation (2x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Current 1h candle properties
        curr_open = open_1h[i // 4] if i >= 4 else open_1h[0]  # Approximate 1h index
        curr_close = close_1h[i // 4] if i >= 4 else close_1h[0]
        curr_body = np.abs(curr_close - curr_open)
        
        # Strong candle: body > 1.5x average
        strong_candle = curr_body > 1.5 * body_mean_aligned
        
        # Bullish/bearish candle
        bullish_candle = curr_close > curr_open
        bearish_candle = curr_close < curr_open
        
        if position == 0:  # No position - look for mean reversion entries
            # Near 1h low with bullish rejection
            if (low[i] <= low_1d[i // 4] + 0.1 * atr_1h[i] and 
                bullish_candle and strong_candle and volume_confirm):
                position = 1
                signals[i] = position_size
            # Near 1h high with bearish rejection
            elif (high[i] >= high_1d[i // 4] - 0.1 * atr_1h[i] and 
                  bearish_candle and strong_candle and volume_confirm):
                position = -1
                signals[i] = -position_size
        elif position == 1:  # Long position - exit on reversal or time
            # Exit on bearish rejection at resistance
            if (high[i] >= high_1d[i // 4] - 0.1 * atr_1h[i] and 
                bearish_candle and strong_candle):
                position = 0
                signals[i] = 0.0
            # Time-based exit: hold max 4 periods (16 hours)
            elif i % 16 >= 15:  # Exit near end of 4h cycle
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit on reversal or time
            # Exit on bullish rejection at support
            if (low[i] <= low_1d[i // 4] + 0.1 * atr_1h[i] and 
                bullish_candle and strong_candle):
                position = 0
                signals[i] = 0.0
            # Time-based exit: hold max 4 periods (16 hours)
            elif i % 16 >= 15:  # Exit near end of 4h cycle
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Volume_Price_Action_Range_Breakout"
timeframe = "4h"
leverage = 1.0