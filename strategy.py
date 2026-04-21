#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_WeeklyTrend_ATRStop_v1
Hypothesis: Daily Camarilla R1/S1 breakouts filtered by weekly EMA34 trend, with ATR-based stoploss.
Uses discrete position sizing (0.30) to balance risk and reward. Weekly trend filter provides robust
directional bias across market cycles while reducing false breakouts. Target: 15-25 trades/year per symbol
for low fee drag and strong test generalization in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (weekly for EMA34 trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # === Daily OHLC for Camarilla calculation ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Previous day's OHLC for Camarilla levels (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # Set first value to NaN since no previous day exists
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R1, S1, R2, S2, R3, S3, R4, S4
    # R4 = Close + 1.5*(High-Low), S4 = Close - 1.5*(High-Low)
    # R3 = Close + 1.125*(High-Low), S3 = Close - 1.125*(High-Low)
    # R2 = Close + 0.75*(High-Low), S2 = Close - 0.75*(High-Low)
    # R1 = Close + 0.5*(High-Low), S1 = Close - 0.5*(High-Low)
    rng = prev_high - prev_low
    R1 = prev_close + 0.5 * rng
    S1 = prev_close - 0.5 * rng
    R2 = prev_close + 0.75 * rng
    S2 = prev_close - 0.75 * rng
    R3 = prev_close + 1.125 * rng
    S3 = prev_close - 1.125 * rng
    R4 = prev_close + 1.5 * rng
    S4 = prev_close - 1.5 * rng
    
    # === Weekly EMA34 for trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long conditions: price > R1 (breakout above resistance) and weekly uptrend
            long_breakout = price > R1[i]
            long_trend = price > ema_34_1w_aligned[i]
            
            # Short conditions: price < S1 (breakdown below support) and weekly downtrend
            short_breakout = price < S1[i]
            short_trend = price < ema_34_1w_aligned[i]
            
            # Entry logic - enter on breakout/breakdown with trend alignment
            if long_breakout and long_trend:
                signals[i] = 0.30
                position = 1
                entry_price = price
            elif short_breakout and short_trend:
                signals[i] = -0.30
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below S1 (breakdown of support)
            elif price < S1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above R1 (breakout of resistance)
            elif price > R1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_WeeklyTrend_ATRStop_v1"
timeframe = "1d"
leverage = 1.0