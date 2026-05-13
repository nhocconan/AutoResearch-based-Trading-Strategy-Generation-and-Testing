#!/usr/bin/env python3
"""
4h_Momentum_Confluence_Strategy
Hypothesis: Combining momentum (MACD), trend (ADX), and volume confirmation on 4H timeframe
creates robust entries in both bull and bear markets. Uses ETH/BTC-proven indicators with
strict entry conditions to limit trades (~20-35/year) and avoid fee drag. Exits on momentum
reversal or trend exhaustion. Position size 0.25 balances risk and return.
"""

name = "4h_Momentum_Confluence_Strategy"
timeframe = "4h"
leverage = 1.0

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
    
    # MACD calculation (12,26,9)
    close_s = pd.Series(close)
    ema12 = close_s.ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = close_s.ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # ADX calculation (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (pd.Series(atr).rolling(window=14, min_periods=14).sum().values + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (pd.Series(atr).rolling(window=14, min_periods=14).sum().values + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Trend filter: EMA50 on 4H
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: MACD bullish crossover + ADX > 20 (trending) + above EMA50 + volume
            if (macd_line[i] > signal_line[i] and 
                macd_line[i-1] <= signal_line[i-1] and
                adx[i] > 20 and 
                close[i] > ema50[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: MACD bearish crossover + ADX > 20 + below EMA50 + volume
            elif (macd_line[i] < signal_line[i] and 
                  macd_line[i-1] >= signal_line[i-1] and
                  adx[i] > 20 and 
                  close[i] < ema50[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: MACD bearish crossover or ADX weakens (<20) or price below EMA50
            if (macd_line[i] < signal_line[i] and 
                macd_line[i-1] >= signal_line[i-1]) or \
               adx[i] < 20 or \
               close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: MACD bullish crossover or ADX weakens or price above EMA50
            if (macd_line[i] > signal_line[i] and 
                macd_line[i-1] <= signal_line[i-1]) or \
               adx[i] < 20 or \
               close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals