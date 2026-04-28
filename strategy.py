#!/usr/bin/env python3
"""
4h_TrueRangeBreakout_12hATR_Volume_Confirmation
Hypothesis: Breakouts beyond true range with 12h ATR filter and volume confirmation capture strong momentum in both bull and bear markets.
Uses tight entry conditions (breakout > 12h ATR) to limit trades (target 30-50/year) and reduce fee drag. Works by filtering out low-volatility breakouts
and requiring volume surge, avoiding whipsaws in ranging conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ATR and trend
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate True Range and ATR(14) on 12h
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_12h = tr.rolling(window=14, min_periods=14).mean().values
    
    # 12h EMA25 for trend filter
    ema_25_12h = pd.Series(df_12h['close']).ewm(span=25, adjust=False, min_periods=25).mean().values
    
    # Align all higher timeframe data to 4h
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    ema_25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_25_12h)
    
    # Trend filter: price > EMA25 = bullish, < EMA25 = bearish
    h12_uptrend = close > ema_25_12h_aligned
    h12_downtrend = close < ema_25_12h_aligned
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_12h_aligned[i]) or np.isnan(ema_25_12h_aligned[i]) or 
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: breakout beyond 12h ATR + trend alignment + volume surge
        # Long: price breaks above high + ATR + trend + volume
        long_entry = (close[i] > (high[i-1] + atr_12h_aligned[i]) and 
                     h12_uptrend[i] and 
                     volume_surge[i])
        
        # Short: price breaks below low - ATR + trend + volume
        short_entry = (close[i] < (low[i-1] - atr_12h_aligned[i]) and 
                      h12_downtrend[i] and 
                      volume_surge[i])
        
        # Exit on opposite breakout with volume surge
        long_exit = close[i] < (low[i-1] - atr_12h_aligned[i]) and volume_surge[i]
        short_exit = close[i] > (high[i-1] + atr_12h_aligned[i]) and volume_surge[i]
        
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

name = "4h_TrueRangeBreakout_12hATR_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0