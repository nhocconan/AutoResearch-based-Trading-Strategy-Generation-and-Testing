#!/usr/bin/env python3
# 4H_Three_Month_High_Low_Breakout_With_12H_Trend
# Hypothesis: Breakouts from 3-month (12-week) high/low levels with 12-hour trend filter on 4h timeframe.
# Uses monthly extremes as structural support/resistance, filtered by 12h EMA trend and volume confirmation.
# Works in bull markets (breakouts above 3m high in uptrend) and bear markets (breakdowns below 3m low in downtrend).
# Low-frequency signals reduce fee drag; structural levels provide edge in ranging markets.

name = "4H_Three_Month_High_Low_Breakout_With_12H_Trend"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 12h data for 3-month high/low calculation (12 weeks = 84 periods)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 84:  # Need 12 weeks of data
        return np.zeros(n)
    
    # Calculate 3-month (84-period) rolling high and low on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Use pandas rolling for efficiency with min_periods
    high_3m = pd.Series(high_12h).rolling(window=84, min_periods=84).max().values
    low_3m = pd.Series(low_12h).rolling(window=84, min_periods=84).min().values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 12h indicators to 4h timeframe
    high_3m_aligned = align_htf_to_ltf(prices, df_12h, high_3m)
    low_3m_aligned = align_htf_to_ltf(prices, df_12h, low_3m)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: current volume > 1.5x average volume (30-period)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # Volatility filter: avoid extremely low volatility (ATR > 0.3% of price)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    vol_filter = atr > 0.003 * close  # ATR > 0.3% of price
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(84, 50, 30)  # Ensure we have all required data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(high_3m_aligned[i]) or np.isnan(low_3m_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(vol_filter[i]) or not vol_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation (1.5x average volume)
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above 3-month high + uptrend (price > 12h EMA50) + volume
            if (close[i] > high_3m_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and   # Uptrend filter
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 3-month low + downtrend (price < 12h EMA50) + volume
            elif (close[i] < low_3m_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and   # Downtrend filter
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit conditions:
            # 1. Price returns to opposite extreme (mean reversion)
            # 2. Trend reversal (price crosses 12h EMA50 in opposite direction)
            trend_reversal = (position == 1 and close[i] < ema_50_12h_aligned[i]) or \
                           (position == -1 and close[i] > ema_50_12h_aligned[i])
            
            if trend_reversal:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals