#!/usr/bin/env python3
"""
1h_4d_Camarilla_Pullback_v1
Hypothesis: On 1h timeframe, buy pullbacks to Camarilla support levels during strong uptrends (4h EMA50) and sell rallies to resistance during strong downtrends (4h EMA50). Uses 1d volatility filter (ATR ratio) to avoid choppy markets. Designed for 15-30 trades/year by requiring trend alignment, volatility filter, and precise Camarilla levels. Works in bull markets via long pullbacks and in bear markets via short rallies.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_Camarilla_Pullback_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.abs(high_1d[0] - low_1d[0])
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # Calculate Camarilla levels from previous day
    def calculate_camarilla(h_prev, l_prev, c_prev):
        range_val = h_prev - l_prev
        if range_val <= 0:
            return c_prev, c_prev, c_prev, c_prev
        multiplier = range_val * 1.1 / 12
        l3 = c_prev + multiplier * 1.1
        l4 = c_prev + multiplier * 1.5
        h3 = c_prev - multiplier * 1.1
        h4 = c_prev - multiplier * 1.5
        return l3, l4, h3, h4
    
    # Shift to get previous day's OHLC
    h_prev = np.roll(high, 24)  # 24 hours in previous day
    l_prev = np.roll(low, 24)
    c_prev = np.roll(close, 24)
    h_prev[0:24] = h_prev[24]  # fill first day
    l_prev[0:24] = l_prev[24]
    c_prev[0:24] = c_prev[24]
    
    l3, l4, h3, h4 = calculate_camarilla(h_prev, l_prev, c_prev)
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_ma_1d_aligned[i]) or np.isnan(l3[i]) or np.isnan(l4[i]) or
            np.isnan(h3[i]) or np.isnan(h4[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Session filter
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Volatility filter: avoid high volatility (ATR > 1.5x MA)
        vol_filter = atr_1d_aligned[i] <= atr_ma_1d_aligned[i] * 1.5
        
        # Trend filter from 4h EMA50
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        # Camarilla levels
        near_l3 = abs(close[i] - l3[i]) / l3[i] < 0.003  # within 0.3%
        near_l4 = abs(close[i] - l4[i]) / l4[i] < 0.003
        near_h3 = abs(close[i] - h3[i]) / h3[i] < 0.003
        near_h4 = abs(close[i] - h4[i]) / h4[i] < 0.003
        
        # Entry conditions
        long_entry = in_session and vol_filter and uptrend and (near_l3 or near_l4)
        short_entry = in_session and vol_filter and downtrend and (near_h3 or near_h4)
        
        # Exit conditions: opposite Camarilla level or trend reversal
        long_exit = near_h3 or near_h4 or not uptrend
        short_exit = near_l3 or near_l4 or not downtrend
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals