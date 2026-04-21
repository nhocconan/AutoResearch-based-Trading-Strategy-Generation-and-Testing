#!/usr/bin/env python3
"""
1d_KAMA_Regime_Filter_DonchianExit
Hypothesis: Daily KAMA direction filtered by weekly EMA34 trend and Bollinger Bandwidth regime (choppy < 0.05 = mean revert, trending > 0.08 = trend follow).
Long when KAMA rising and price > weekly EMA34 and BBWidth > 0.08; short when KAMA falling and price < weekly EMA34 and BBWidth > 0.08.
Mean reversion in chop: long at lower Bollinger Band, short at upper Bollinger Band when BBWidth < 0.05.
ATR-based stoploss (2.0x) and discrete sizing (0.25).
Designed to work in both bull and bear markets via regime adaptation.
Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (weekly for EMA trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # === Weekly EMA34 for trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily KAMA (ER=10, SC=2/30+2/2+2) ===
    close = prices['close'].values
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    direction = np.abs(np.diff(close, prepend=close[0]))
    er = np.where(volatility > 0, direction / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === Daily Bollinger Bands (20, 2) for regime and mean reversion ===
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle  # normalized bandwidth
    
    # === Daily ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # warmup for KAMA, BB, ATR
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(kama[i]) 
            or np.isnan(bb_middle[i]) or np.isnan(bb_width[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_now = kama[i]
        kama_prev = kama[i-1]
        ema_trend = ema_34_1w_aligned[i]
        bb_Width = bb_width[i]
        bb_up = bb_upper[i]
        bb_low = bb_lower[i]
        
        # KAMA direction: rising or falling
        kama_rising = kama_now > kama_prev
        kama_falling = kama_now < kama_prev
        
        if position == 0:
            if bb_Width > 0.08:  # trending regime
                # Trend following: follow KAMA direction aligned with weekly trend
                long_condition = kama_rising and (price > ema_trend)
                short_condition = kama_falling and (price < ema_trend)
                
                if long_condition:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif short_condition:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            elif bb_Width < 0.05:  # choppy regime: mean reversion at Bollinger Bands
                long_condition = price <= bb_low
                short_condition = price >= bb_up
                
                if long_condition:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif short_condition:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend regime: exit if KAMA turns or price breaks weekly trend
            elif bb_Width > 0.08:
                if not kama_rising or price < ema_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            # Chop regime: exit at mean reversion to middle Bollinger Band
            else:
                if price >= bb_middle:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend regime: exit if KAMA turns or price breaks weekly trend
            elif bb_Width > 0.08:
                if not kama_falling or price > ema_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            # Chop regime: exit at mean reversion to middle Bollinger Band
            else:
                if price <= bb_middle:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Regime_Filter_DonchianExit"
timeframe = "1d"
leverage = 1.0