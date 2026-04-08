#!/usr/bin/env python3
# 12h_price_channel_breakout_1d_trend
# Hypothesis: Price channel breakouts on 12h timeframe filtered by 1d trend (EMA crossover) and volume confirmation.
# Works in bull markets (breakouts above upper channel) and bear markets (breakdowns below lower channel).
# Target: 20-40 trades/year on 12h timeframe with strict entry conditions to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_price_channel_breakout_1d_trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h price channel (Donchian 20-period)
    period = 20
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # 1d EMA for trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema100 = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align daily EMA to 12h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    ema100_aligned = align_htf_to_ltf(prices, df_1d, ema100)
    
    # Volume confirmation: 12h volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure EMA100 and ATR are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(ema100_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.0 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: Price breaks below lower channel OR stoploss hit
            if close[i] < lowest_low[i] or close[i] < ema50_aligned[i] - 3.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above upper channel OR stoploss hit
            if close[i] > highest_high[i] or close[i] > ema50_aligned[i] + 3.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above upper channel with volume surge and bullish trend
            if close[i] > highest_high[i] and vol_surge and ema50_aligned[i] > ema100_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below lower channel with volume surge and bearish trend
            elif close[i] < lowest_low[i] and vol_surge and ema50_aligned[i] < ema100_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals