#!/usr/bin/env python3
# 1h_4h_1d_camarilla_breakout_trend_v1
# Strategy: Camarilla pivot breakout with 4h trend filter and 1d volatility filter
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: Price breaking Camarilla H3/L3 levels with 4h trend alignment and 1d volatility filter
# captures institutional breakouts while avoiding chop. Works in bull (breakouts up) and bear
# (breakouts down) by following 4h trend direction. Limited trades via strict confluence.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_breakout_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_4h_up = ema_4h > np.roll(ema_4h, 1)  # Rising EMA = uptrend
    trend_4h_up[0] = False
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_up)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_filter_1d = atr_1d > atr_ma_1d  # Above average volatility
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(trend_4h_aligned[i]) or np.isnan(vol_filter_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Skip if outside session
        if not session_filter[i]:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Calculate Camarilla levels for current hour using previous hour's data
        if i < 2:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
            
        ph = high[i-1]  # Previous hour high
        pl = low[i-1]   # Previous hour low
        pc = close[i-1] # Previous hour close
        
        range_ = ph - pl
        if range_ <= 0:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
            
        # Camarilla levels
        h3 = pc + (range_ * 1.1 / 4)
        l3 = pc - (range_ * 1.1 / 4)
        h4 = pc + (range_ * 1.1 / 2)
        l4 = pc - (range_ * 1.1 / 2)
        
        # Entry conditions
        long_breakout = close[i] > h3 and trend_4h_aligned[i] and vol_filter_aligned[i]
        short_breakout = close[i] < l3 and not trend_4h_aligned[i] and vol_filter_aligned[i]
        
        # Exit conditions: reverse signal or opposite Camarilla level touch
        exit_long = position == 1 and (close[i] < l3 or not trend_4h_aligned[i])
        exit_short = position == -1 and (close[i] > h3 or trend_4h_aligned[i])
        
        # Trading logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals