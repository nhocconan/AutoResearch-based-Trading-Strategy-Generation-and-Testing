#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based stoploss.
# Long when price breaks above 20-period Donchian high AND 1d EMA50 uptrend.
# Short when price breaks below 20-period Donchian low AND 1d EMA50 downtrend.
# Uses ATR(14) stoploss: exit long if price < highest_since_entry - 2.0*ATR(14),
# exit short if price > lowest_since_entry + 2.0*ATR(14).
# Discrete position sizing (0.25) minimizes fee churn. Target: 19-50 trades/year on 4h timeframe.
# Donchian channels provide clear structure, EMA50 filters counter-trend noise, ATR stops manage risk.

name = "4h_Donchian20_Breakout_1dEMA50_ATR_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d EMA50 for trend filter (loaded once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 14-period ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup for EMA50, ATR, and Donchian channels
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        
        # Trend filter: 1d EMA50 direction
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Donchian breakout signals (using current bar's high/low vs previous bar's channels)
        breakout_up = curr_high > donchian_high[i-1]
        breakout_down = curr_low < donchian_low[i-1]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout up AND uptrend
            if breakout_up and uptrend:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            # Short: Breakout down AND downtrend
            elif breakout_down and downtrend:
                signals[i] = -0.25
                position = -1
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_close > highest_since_entry:
                highest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR trend reversal
            stop_price = highest_since_entry - 2.0 * curr_atr
            if curr_close < stop_price or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR trend reversal
            stop_price = lowest_since_entry + 2.0 * curr_atr
            if curr_close > stop_price or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals