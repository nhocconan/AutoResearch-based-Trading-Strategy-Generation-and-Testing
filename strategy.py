#!/usr/bin/env python3
"""
1d_1w_Price_Action_Volume_Regime_Strategy
Hypothesis: Daily price action combined with weekly trend filter, volume confirmation, and chop regime filter captures high-probability swing moves.
Uses daily breakout of prior day's high/low with volume > 1.5x 20-day average, weekly EMA trend filter, and Chop index < 61.8 for trending markets.
Works in bull markets via breakouts above daily high and in bear markets via breakdowns below daily low. Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for stop loss and Chop calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop index calculation (14-period)
    def calculate_chop(high, low, close, window=14):
        atr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum()
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max()
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min()
        chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(window)
        return chop.fillna(50).values  # neutral when undefined
    
    chop = calculate_chop(high, low, close, 14)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # Daily breakout levels: previous day's high/low
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan  # first day has no previous
    prev_low[0] = np.nan
    
    # Weekly EMA trend filter (using 1w data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        ema21_1w = np.full(len(prices), np.nan)
    else:
        close_1w = df_1w['close'].values
        ema21_1w_raw = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
        ema21_1w = align_htf_to_ltf(prices, df_1w, ema21_1w_raw)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or 
            np.isnan(ema21_1w[i]) or np.isnan(chop[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (Chop < 61.8)
        is_trending = chop[i] < 61.8
        
        # Long signal: break above previous day's high with volume expansion and weekly uptrend
        long_signal = (high[i] > prev_high[i] and 
                      volume_expansion[i] and 
                      close[i] > ema21_1w[i] and 
                      is_trending)
        
        # Short signal: break below previous day's low with volume expansion and weekly downtrend
        short_signal = (low[i] < prev_low[i] and 
                       volume_expansion[i] and 
                       close[i] < ema21_1w[i] and 
                       is_trending)
        
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1d_1w_Price_Action_Volume_Regime_Strategy"
timeframe = "1d"
leverage = 1.0