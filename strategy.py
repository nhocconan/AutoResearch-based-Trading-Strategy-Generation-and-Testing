#!/usr/bin/env python3
"""
4H_382_Camarilla_Reversal
Hypothesis: On 4-hour timeframe, long when price touches Camarilla S3 level with bullish divergence and volume confirmation, short when price touches R3 level with bearish divergence. Uses 1-day trend filter to avoid counter-trend trades. Designed for low trade frequency (<30/year) to minimize fee drift and work in both bull/bear markets via mean reversion at key levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day using previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels
    R3 = prev_close + (prev_range * 1.1 / 4)
    S3 = prev_close - (prev_range * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    
    # 1-day trend filter: EMA34 > EMA50 = uptrend
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema34_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    daily_uptrend = ema34_4h > ema50_4h
    
    # RSI divergence (14-period)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Bullish divergence: price makes lower low, RSI makes higher low
    # Bearish divergence: price makes higher high, RSI makes lower high
    lookback = 5
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        if low[i] < low[i-lookback] and rsi_values[i] > rsi_values[i-lookback]:
            bullish_div[i] = True
        if high[i] > high[i-lookback] and rsi_values[i] < rsi_values[i-lookback]:
            bearish_div[i] = True
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_4h[i]) or np.isnan(S3_4h[i]) or 
            np.isnan(ema34_4h[i]) or np.isnan(ema50_4h[i]) or
            np.isnan(rsi_values[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = (close[i] <= S3_4h[i] * 1.002) and bullish_div[i] and volume_surge[i] and daily_uptrend[i]
        short_entry = (close[i] >= R3_4h[i] * 0.998) and bearish_div[i] and volume_surge[i] and not daily_uptrend[i]
        
        # Exit on opposite touch of Camarilla level
        long_exit = close[i] >= R3_4h[i] * 0.998
        short_exit = close[i] <= S3_4h[i] * 1.002
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4H_382_Camarilla_Reversal"
timeframe = "4h"
leverage = 1.0