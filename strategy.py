#!/usr/bin/env python3
"""
6h_OrderBlock_OrderFlow_Imbalance
Hypothesis: Combines order block detection (bullish/bearish OB from swing highs/lows) 
with volume imbalance (delta volume) to identify high-probability reversal zones. 
Works in both bull and bear markets by trading mean reversion from institutional 
order blocks when retail momentum exhausts. Uses 12h trend filter to avoid 
counter-trend trades in strong trends. Targets 15-30 trades/year for low fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate swing points (using 5-bar window)
    # Bullish swing low: low[i] is lowest in window
    # Bearish swing high: high[i] is highest in window
    window = 5
    bullish_swing = np.zeros(n, dtype=bool)
    bearish_swing = np.zeros(n, dtype=bool)
    
    for i in range(window, n - window):
        if low[i] == np.min(low[i-window:i+window+1]):
            bullish_swing[i] = True
        if high[i] == np.max(high[i-window:i+window+1]):
            bearish_swing[i] = True
    
    # Identify order blocks: last opposite candle before swing
    # Bullish OB: last bearish candle before bullish swing low
    # Bearish OB: last bullish candle before bearish swing high
    bullish_ob = np.zeros(n, dtype=bool)
    bearish_ob = np.zeros(n, dtype=bool)
    
    for i in range(window, n):
        if bullish_swing[i]:
            # Look back for bearish candle
            for j in range(i-1, max(i-10, window-1), -1):
                if close[j] < open_[j]:  # bearish candle
                    bullish_ob[j] = True
                    break
        if bearish_swing[i]:
            # Look back for bullish candle
            for j in range(i-1, max(i-10, window-1), -1):
                if close[j] > open_[j]:  # bullish candle
                    bearish_ob[j] = True
                    break
    
    # Need open prices
    open_ = prices['open'].values
    
    # Volume imbalance: delta volume (buy vol - sell vol) approximated by 
    # volume * sign(close - open)
    delta_volume = volume * np.where(close >= open_, 1, -1)
    # Volume imbalance signal: look for divergence (price makes new low/high but 
    # delta volume doesn't confirm)
    vol_ma = pd.Series(delta_volume).rolling(window=20, min_periods=20).mean().values
    
    # Bullish imbalance: price makes new low but delta volume > ma (hidden bullish)
    # Bearish imbalance: price makes new high but delta volume < ma (hidden bearish)
    lowest_low = pd.Series(low).rolling(window=50, min_periods=50).min().values
    highest_high = pd.Series(high).rolling(window=50, min_periods=50).max().values
    
    bullish_imbalance = (low <= lowest_low) & (delta_volume > vol_ma)
    bearish_imbalance = (high >= highest_high) & (delta_volume < vol_ma)
    
    # Align order blocks and imbalances
    bullish_ob_aligned = align_htf_to_ltf(prices, pd.DataFrame({'dummy': bullish_ob}), bullish_ob.astype(float)) > 0.5
    bearish_ob_aligned = align_htf_to_ltf(prices, pd.DataFrame({'dummy': bearish_ob}), bearish_ob.astype(float)) > 0.5
    bullish_imb_aligned = align_htf_to_ltf(prices, pd.DataFrame({'dummy': bullish_imbalance}), bullish_imbalance.astype(float)) > 0.5
    bearish_imb_aligned = align_htf_to_ltf(prices, pd.DataFrame({'dummy': bearish_imbalance}), bearish_imbalance.astype(float)) > 0.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Entry conditions: mean reversion from order blocks with volume imbalance
        long_entry = bullish_ob_aligned[i] and bullish_imb_aligned[i] and downtrend
        short_entry = bearish_ob_aligned[i] and bearish_imb_aligned[i] and uptrend
        
        # Exit conditions: price reaches opposite order block or trend strengthens
        long_exit = bearish_ob_aligned[i] or (close[i] > ema_50_12h_aligned[i] * 1.02)
        short_exit = bullish_ob_aligned[i] or (close[i] < ema_50_12h_aligned[i] * 0.98)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_OrderBlock_OrderFlow_Imbalance"
timeframe = "6h"
leverage = 1.0