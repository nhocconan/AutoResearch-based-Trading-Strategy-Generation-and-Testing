#!/usr/bin/env python3
"""
4h_KAMA_Trend_Breakout_1dTrend_Filter
Hypothesis: On 4h timeframe, KAMA direction determines trend (bullish when price > KAMA, bearish when price < KAMA). Enter long when price breaks above 4h Donchian high(20) in bullish trend, short when price breaks below Donchian low(20) in bearish trend. Use 1d EMA34 trend filter to only trade in direction of daily trend. Require volume > 1.5x 20-period average for confirmation. Exits when price crosses KAMA in opposite direction. Targets 20-40 trades/year by requiring trend alignment, breakout, and volume confirmation. Works in bull markets via long trades in uptrends and bear markets via short trades in downtrends.
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
    
    # 4h KAMA for trend
    price_change = abs(np.diff(close, prepend=close[0]))
    direction = np.abs(np.diff(close, 10))
    volatility = np.sum(np.abs(np.diff(close, 1)), axis=0) if len(close.shape) > 1 else np.abs(np.diff(close, 1))
    if len(volatility.shape) == 0:
        volatility = np.full_like(close, np.abs(np.diff(close, 1)))
    volatility = np.where(volatility == 0, 1e-10, volatility)
    er = direction / volatility
    er = np.where(np.isnan(er), 0, er)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    sc = np.where(np.isnan(sc), 0.01, sc)
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 4h Donchian channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Trend alignment: price relative to KAMA and 1d EMA34
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        daily_bullish = close[i] > ema_34_1d_aligned[i]
        daily_bearish = close[i] < ema_34_1d_aligned[i]
        
        # Entry conditions
        long_entry = (close[i] > donch_high[i] and 
                     kama_bullish and 
                     daily_bullish and 
                     volume_surge[i])
        
        short_entry = (close[i] < donch_low[i] and 
                      kama_bearish and 
                      daily_bearish and 
                      volume_surge[i])
        
        # Exit when price crosses KAMA in opposite direction
        long_exit = close[i] < kama[i] and position == 1
        short_exit = close[i] > kama[i] and position == -1
        
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

name = "4h_KAMA_Trend_Breakout_1dTrend_Filter"
timeframe = "4h"
leverage = 1.0