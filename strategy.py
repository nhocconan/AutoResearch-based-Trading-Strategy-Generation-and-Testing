#!/usr/bin/env python3

# 4h_triple_signal_confluence_v1
# Hypothesis: Combines Donchian breakout, RSI momentum, and volume confirmation with 12h trend filter.
# Designed for low trade frequency (<30/year) to minimize fee drag while capturing strong trends.
# Works in both bull and bear markets by requiring multiple confluence factors before entry.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_triple_signal_confluence_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h trend filter - load once before loop
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h data for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 4h indicators
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(rsi[i]) or np.isnan(avg_volume[i]) or np.isnan(ema50_12h_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # 12h trend filter
        trend_up = close[i] > ema50_12h_aligned[i]
        trend_down = close[i] < ema50_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price re-enters Donchian channel or RSI overbought
            if close[i] < highest_high[i] or rsi[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price re-enters Donchian channel or RSI oversold
            if close[i] > lowest_low[i] or rsi[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            if volume_ok:
                # Long entry: Donchian breakout + RSI momentum + 12h uptrend
                if (trend_up and 
                    close[i] > highest_high[i] and 
                    close[i-1] <= highest_high[i-1] and
                    rsi[i] > 50 and rsi[i] < 70):
                    position = 1
                    signals[i] = 0.25
                # Short entry: Donchian breakdown + RSI momentum + 12h downtrend
                elif (trend_down and 
                      close[i] < lowest_low[i] and 
                      close[i-1] >= lowest_low[i-1] and
                      rsi[i] < 50 and rsi[i] > 30):
                    position = -1
                    signals[i] = -0.25
    
    return signals