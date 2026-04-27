#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly EMA(8) and EMA(21) for trend direction
    ema_8_1w = pd.Series(close_1w).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_8_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_8_1w)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate daily support/resistance levels for confluence
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 6-period ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=6, min_periods=6).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_8_1w_aligned[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: EMA8 > EMA21 for uptrend, EMA8 < EMA21 for downtrend
        weekly_uptrend = ema_8_1w_aligned[i] > ema_21_1w_aligned[i]
        weekly_downtrend = ema_8_1w_aligned[i] < ema_21_1w_aligned[i]
        
        # Volatility filter: current ATR above 50% of its 50-period average
        atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean()
        atr_ma_val = atr_ma[i] if i < len(atr_ma) and not np.isnan(atr_ma[i]) else atr[i]
        volatility_filter = atr[i] > 0.5 * atr_ma_val if not np.isnan(atr_ma_val) else True
        
        # Entry conditions
        long_condition = weekly_uptrend and volatility_filter
        short_condition = weekly_downtrend and volatility_filter
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal on weekly timeframe
        elif position == 1 and not weekly_uptrend:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not weekly_downtrend:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_EMA8_EMA21_1wTrend_VolatilityFilter"
timeframe = "6h"
leverage = 1.0