#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout_Volume_Confirmation_v5
Hypothesis: Camarilla pivot levels from daily timeframe act as strong support/resistance.
Breakout above H3 or below L3 with volume expansion (>1.5x 20-period average) and 
ADX trend filter (>25) captures high-probability trend continuation moves.
Works in bull markets via breakouts above H3 and in bear markets via breakdowns below L3.
Uses 4h timeframe with 1d HTF for pivot calculation. Target: 20-35 trades/year per symbol.
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
    
    # Calculate ATR for ADX
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    # Calculate ADX (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr_ma = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / tr_ma
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / tr_ma
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        # Not enough daily data
        H3 = np.full(n, np.nan)
        L3 = np.full(n, np.nan)
    else:
        # Calculate Camarilla levels from previous day's OHLC
        prev_close = df_1d['close'].shift(1).values
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_range = prev_high - prev_low
        
        # Camarilla levels: H3 = close + range * 1.1/4, L3 = close - range * 1.1/4
        H3_raw = prev_close + prev_range * 1.1 / 4
        L3_raw = prev_close - prev_range * 1.1 / 4
        
        # Align to 4h timeframe
        H3 = align_htf_to_ltf(prices, df_1d, H3_raw)
        L3 = align_htf_to_ltf(prices, df_1d, L3_raw)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or 
            np.isnan(adx[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when ADX > 25 (trending market)
        is_trending = adx[i] > 25
        
        # Long signal: break above H3 with volume expansion and trending market
        long_signal = (close[i] > H3[i] and 
                      volume_expansion[i] and 
                      is_trending)
        
        # Short signal: break below L3 with volume expansion and trending market
        short_signal = (close[i] < L3[i] and 
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

name = "4h_1d_Camarilla_Pivot_Breakout_Volume_Confirmation_v5"
timeframe = "4h"
leverage = 1.0