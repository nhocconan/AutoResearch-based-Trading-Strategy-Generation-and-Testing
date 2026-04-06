#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour RSI(14) mean reversion with 4-hour ADX(14) regime filter and 1-day EMA200 trend filter.
# Uses 4h ADX to identify trending (ADX>25) vs ranging (ADX<20) markets.
# In ranging markets (ADX<20): RSI<30 long, RSI>70 short.
# In trending markets (ADX>25): RSI<40 long in uptrend (price>EMA200), RSI>60 short in downtrend (price<EMA200).
# Includes session filter (08-20 UTC) to avoid low-liquidity hours.
# Designed for 1h timeframe to target 60-150 trades over 4 years with strict entry conditions.

name = "1h_rsi14_adx14_ema200_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 1-day EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema_200_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_200_1d[i] = (close_1d[i] * 2 / 201) + (ema_200_1d[i-1] * 199 / 201)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 4-hour ADX(14) for regime filter
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr = np.zeros(len(high_4h))
    tr[0] = high_4h[0] - low_4h[0]
    for i in range(1, len(high_4h)):
        tr[i] = max(high_4h[i] - low_4h[i], abs(high_4h[i] - close_4h[i-1]), abs(low_4h[i] - close_4h[i-1]))
    
    # +DM and -DM
    plus_dm = np.zeros(len(high_4h))
    minus_dm = np.zeros(len(high_4h))
    for i in range(1, len(high_4h)):
        up_move = high_4h[i] - high_4h[i-1]
        down_move = low_4h[i-1] - low_4h[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smoothed values (Wilder's smoothing)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr14 = wilder_smooth(tr, 14)
    plus_dm14 = wilder_smooth(plus_dm, 14)
    minus_dm14 = wilder_smooth(minus_dm, 14)
    
    # DI+ and DI-
    plus_di14 = np.full_like(tr14, np.nan)
    minus_di14 = np.full_like(tr14, np.nan)
    for i in range(len(tr14)):
        if not np.isnan(tr14[i]) and tr14[i] != 0:
            plus_di14[i] = (plus_dm14[i] / tr14[i]) * 100
            minus_di14[i] = (minus_dm14[i] / tr14[i]) * 100
    
    # DX and ADX
    dx = np.full_like(tr14, np.nan)
    for i in range(len(tr14)):
        if not np.isnan(plus_di14[i]) and not np.isnan(minus_di14[i]):
            di_sum = plus_di14[i] + minus_di14[i]
            if di_sum != 0:
                dx[i] = abs(plus_di14[i] - minus_di14[i]) / di_sum * 100
    
    adx_14 = wilder_smooth(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx_14)
    
    # 1-hour RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    def rsi_wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    avg_gain = rsi_wilder_smooth(gain, 14)
    avg_loss = rsi_wilder_smooth(loss, 14)
    rs = np.full_like(avg_gain, np.nan)
    for i in range(len(avg_gain)):
        if not np.isnan(avg_loss[i]) and avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
    rsi = np.full_like(avg_gain, np.nan)
    for i in range(len(rs)):
        if not np.isnan(rs[i]):
            rsi[i] = 100 - (100 / (1 + rs[i]))
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):  # Warmup period
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(ema_200_aligned[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        adx = adx_aligned[i]
        rsi_val = rsi[i]
        price = close[i]
        ema200 = ema_200_aligned[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: RSI > 70 or price < EMA200 in uptrend
            if rsi_val > 70 or (adx > 25 and price < ema200):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI < 30 or price > EMA200 in downtrend
            if rsi_val < 30 or (adx > 25 and price > ema200):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries based on regime
            if adx < 20:  # Ranging market
                if rsi_val < 30:
                    signals[i] = 0.20
                    position = 1
                    entry_price = price
                elif rsi_val > 70:
                    signals[i] = -0.20
                    position = -1
                    entry_price = price
                else:
                    signals[i] = 0.0
            elif adx > 25:  # Trending market
                if price > ema200 and rsi_val < 40:  # Uptrend pullback
                    signals[i] = 0.20
                    position = 1
                    entry_price = price
                elif price < ema200 and rsi_val > 60:  # Downtrend pullback
                    signals[i] = -0.20
                    position = -1
                    entry_price = price
                else:
                    signals[i] = 0.0
            else:  # Transition zone (20 <= ADX <= 25)
                signals[i] = 0.0
    
    return signals