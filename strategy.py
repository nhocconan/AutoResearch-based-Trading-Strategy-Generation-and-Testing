#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_Breakout_Volume_Confirmation
Hypothesis: Daily Camarilla pivot levels act as strong support/resistance. 
Breakout above H3 or below L3 with volume > 1.5x 20-period average triggers trend continuation.
Trades only in trending markets (14-period ADX > 25) to avoid whipsaws in ranging markets.
Position size: 0.25. Target: 15-25 trades/year per symbol.
Works in bull markets via breakouts above H3 and in bear markets via breakdowns below L3.
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
    
    # Calculate ADX for trend filter (14-period)
    def calculate_adx(high, low, close, window=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[0], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        tr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum()
        dm_plus_sum = pd.Series(dm_plus).rolling(window=window, min_periods=window).sum()
        dm_minus_sum = pd.Series(dm_minus).rolling(window=window, min_periods=window).sum()
        
        # Directional Indicators
        plus_di = 100 * dm_plus_sum / tr_sum
        minus_di = 100 * dm_minus_sum / tr_sum
        
        # DX and ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = pd.Series(dx).rolling(window=window, min_periods=window).mean()
        return adx.fillna(0).values
    
    adx = calculate_adx(high, low, close, 14)
    
    # Daily Camarilla pivot levels (using previous day's OHLC)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla levels
    range_val = prev_high - prev_low
    h3 = prev_close + range_val * 1.1 / 4
    l3 = prev_close - range_val * 1.1 / 4
    h4 = prev_close + range_val * 1.1 / 2
    l4 = prev_close - range_val * 1.1 / 2
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(prev_close[i]) or np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or 
            np.isnan(h3[i]) or np.isnan(l3[i]) or np.isnan(adx[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when ADX > 25 (trending market)
        is_trending = adx[i] > 25
        
        # Long signal: break above H3 with volume expansion and trending market
        long_signal = (high[i] > h3[i] and 
                      volume_expansion[i] and 
                      is_trending)
        
        # Short signal: break below L3 with volume expansion and trending market
        short_signal = (low[i] < l3[i] and 
                       volume_expansion[i] and 
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

name = "12h_1d_Camarilla_Pivot_Breakout_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0