#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and ATR-based stoploss
# Long: Close > Donchian Upper(20) AND price > 1w EMA50
# Short: Close < Donchian Lower(20) AND price < 1w EMA50
# Exit: Close crosses Donchian Middle(20) OR ATR stoploss hit
# ATR stoploss: 2.5 * ATR(14) from entry price
# Donchian channels provide clear breakout levels that work in trending markets
# 1w EMA50 filter ensures we trade with the higher timeframe trend
# Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe
# Discrete position sizing: 0.25 for long/short, 0.0 for flat to minimize fee churn

name = "1d_Donchian_Breakout_1wEMA50_ATRStop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period)
    # Upper: highest high of last 20 periods
    # Lower: lowest low of last 20 periods
    # Middle: average of upper and lower
    upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_upper = upper_20[i]
        curr_lower = lower_20[i]
        curr_middle = middle_20[i]
        curr_ema_1w = ema_50_1w_aligned[i]
        curr_atr = atr[i]
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry
            stop_price = entry_price - 2.5 * curr_atr
            # Exit conditions: Close below Donchian Middle OR stoploss hit
            if curr_close < curr_middle or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry
            stop_price = entry_price + 2.5 * curr_atr
            # Exit conditions: Close above Donchian Middle OR stoploss hit
            if curr_close > curr_middle or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Close > Donchian Upper AND price > 1w EMA50
            if curr_close > curr_upper and curr_close > curr_ema_1w:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: Close < Donchian Lower AND price < 1w EMA50
            elif curr_close < curr_lower and curr_close < curr_ema_1w:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals