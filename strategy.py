#!/usr/bin/env python3
# 1h_4h1d_CombinedTrend_With_Volume
# Hypothesis: Use 4h EMA21 and 1d EMA50 for trend direction, enter on 1h pullbacks with volume confirmation during active sessions.
# Works in bull via pullbacks in uptrend (4h/1d bullish), and in bear via bounces in downtrend (4h/1d bearish).
# Target: 15-37 trades/year per symbol by using strict trend alignment and volume filters.

name = "1h_4h1d_CombinedTrend_With_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA21 for trend
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1d EMA50 for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR for volatility and trailing stop
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_21_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Trend filters
        bullish_trend = ema_21_4h_aligned[i] > ema_50_1d_aligned[i]
        bearish_trend = ema_21_4h_aligned[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_surge = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0 and in_session:
            # Long: price near 4h EMA21 in bullish alignment with volume
            if bullish_trend and close[i] <= ema_21_4h_aligned[i] * 1.005 and volume_surge:
                signals[i] = 0.20
                position = 1
                highest_high_since_entry = high[i]
            # Short: price near 4h EMA21 in bearish alignment with volume
            elif bearish_trend and close[i] >= ema_21_4h_aligned[i] * 0.995 and volume_surge:
                signals[i] = -0.20
                position = -1
                lowest_low_since_entry = low[i]
        else:
            if position == 1:
                # Update highest high since entry
                if high[i] > highest_high_since_entry:
                    highest_high_since_entry = high[i]
                
                # Trailing stop: exit if price drops 2.5*ATR from highest high
                if close[i] < highest_high_since_entry - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    highest_high_since_entry = 0.0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Update lowest low since entry
                if low[i] < lowest_low_since_entry:
                    lowest_low_since_entry = low[i]
                
                # Trailing stop: exit if price rises 2.5*ATR from lowest low
                if close[i] > lowest_low_since_entry + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    lowest_low_since_entry = 0.0
                else:
                    signals[i] = -0.20
    
    return signals