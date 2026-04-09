#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA(50) trend filter and ATR stoploss
# - Uses 1w EMA(50) for trend direction (long when price > EMA, short when price < EMA)
# - Uses 1d Donchian(20) channels for breakout entries
# - ATR(14) stoploss and time-based exit (max 10 days holding)
# - Fixed position size 0.25 to manage drawdown
# - Target: 15-25 trades/year on 1d timeframe (60-100 total over 4 years)
# - Works in bull markets via breakouts above resistance, in bear via breakdowns below support
# - Weekly trend filter prevents counter-trend trades in choppy markets

name = "1d_1w_donchian_breakout_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Pre-compute 1d Donchian(20) channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian channels: highest high/lowest low of last 20 periods
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Pre-compute 1d ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_bar = 0  # track entry bar for time-based exit
    
    for i in range(lookback, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1w EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions: stoploss, mean reversion, or time-based
            if close[i] < highest_high[i] - 2.0 * atr[i]:  # ATR stop
                position = 0
                entry_bar = 0
                signals[i] = 0.0
            elif close[i] < lowest_low[i]:  # Mean reversion exit (break below Donchian low)
                position = 0
                entry_bar = 0
                signals[i] = 0.0
            elif i - entry_bar >= 10:  # Time-based exit (max 10 days)
                position = 0
                entry_bar = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: stoploss, mean reversion, or time-based
            if close[i] > lowest_low[i] + 2.0 * atr[i]:  # ATR stop
                position = 0
                entry_bar = 0
                signals[i] = 0.0
            elif close[i] > highest_high[i]:  # Mean reversion exit (break above Donchian high)
                position = 0
                entry_bar = 0
                signals[i] = 0.0
            elif i - entry_bar >= 10:  # Time-based exit (max 10 days)
                position = 0
                entry_bar = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout entries in direction of 1w trend
            if uptrend and close[i] > highest_high[i]:  # Break above Donchian high in uptrend
                position = 1
                entry_bar = i
                signals[i] = 0.25
            elif downtrend and close[i] < lowest_low[i]:  # Break below Donchian low in downtrend
                position = -1
                entry_bar = i
                signals[i] = -0.25
    
    return signals