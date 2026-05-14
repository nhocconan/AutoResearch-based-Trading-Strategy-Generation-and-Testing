#!/usr/bin/env python3
# Hypothesis: 1h RSI(14) extreme with 4h Donchian(20) breakout and 1d EMA(50) trend filter.
# Long when: RSI < 30 (oversold) AND price breaks above 4h Donchian upper (20) AND close > 1d EMA(50) (bullish trend).
# Short when: RSI > 70 (overbought) AND price breaks below 4h Donchian lower (20) AND close < 1d EMA(50) (bearish trend).
# Exit when: RSI crosses back above 50 (for long) or below 50 (for short) OR price crosses 1d EMA(50) in opposite direction.
# Uses 4h for structure (Donchian breakout) and 1d for trend filter to reduce noise. RSI extremes provide mean reversion edge in both bull and bear markets.
# Session filter (08-20 UTC) reduces off-hours noise. Target size 0.20 to manage drawdown.
# Expected trades: 20-50/year per symbol (80-200 over 4 years) to stay within fee drag limits for 1h timeframe.

name = "1h_RSIExtreme_4hDonchian20_1dEMA50_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1h Indicators (LTF) ---
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # --- 4h Indicators (HTF) ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian(20)
    donchian_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) - trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # --- Session filter (08-20 UTC) ---
    # prices.index is already DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data or outside session
        if (np.isnan(rsi[i]) or
            np.isnan(donchian_high_20_aligned[i]) or
            np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: RSI < 30 (oversold) AND price > 4h Donchian high (breakout) AND close > 1d EMA50 (bullish trend)
            if (rsi[i] < 30 and 
                close[i] > donchian_high_20_aligned[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: RSI > 70 (overbought) AND price < 4h Donchian low (breakdown) AND close < 1d EMA50 (bearish trend)
            elif (rsi[i] > 70 and 
                  close[i] < donchian_low_20_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 50 (exit oversold) OR close < 1d EMA50 (trend change)
            if (rsi[i] > 50 or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: RSI < 50 (exit overbought) OR close > 1d EMA50 (trend change)
            if (rsi[i] < 50 or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals