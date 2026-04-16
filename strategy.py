#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and 1d regime filter (ADX). 
# Uses 4h EMA200 for trend direction, 1d ADX > 25 for trending regime, 
# and 1h RSI(14) pullback to EMA20 for entry. 
# Works in bull by buying dips in uptrend, works in bear by selling rallies in downtrend.
# Target: 60-150 total trades over 4 years (15-37/year). Size: 0.20.

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 4h EMA200 for trend direction ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # === 1d ADX(14) for regime filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = minus_dm[0] = 0
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_14 = wilders_smoothing(tr, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    plus_di = 100 * plus_dm_smooth / atr_14
    minus_di = 100 * minus_dm_smooth / atr_14
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx_14 = wilders_smoothing(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # === 1h RSI(14) and EMA20 for entry timing ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    def rsi_wilder(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        gain_sum = np.nansum(data[:period])
        loss_sum = np.nansum(-data[:period]) if np.any(data[:period] < 0) else 0
        rs = gain_sum / loss_sum if loss_sum != 0 else 100
        result[period-1] = 100 - (100 / (1 + rs))
        for i in range(period, len(data)):
            gain_val = data[i] if data[i] > 0 else 0
            loss_val = -data[i] if data[i] < 0 else 0
            rs = (gain_sum + gain_val) / (loss_sum + loss_val) if (loss_sum + loss_val) != 0 else 100
            result[i] = 100 - (100 / (1 + rs))
            gain_sum = gain_sum - (gain_sum / period) + gain_val
            loss_sum = loss_sum - (loss_sum / period) + loss_val
        return result
    
    rsi_14 = rsi_wilder(delta, 14)
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === Session filter: 08-20 UTC ===
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 200
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema200_4h_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or
            np.isnan(rsi_14[i]) or
            np.isnan(ema20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        
        # Trend and regime filters
        uptrend = price > ema200_4h_aligned[i]
        downtrend = price < ema200_4h_aligned[i]
        trending = adx_1d_aligned[i] > 25.0
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Uptrend + trending + RSI pullback (<40) to EMA20
            if uptrend and trending and rsi_14[i] < 40 and price > ema20[i]:
                signals[i] = 0.20
                position = 1
                continue
            # Short: Downtrend + trending + RSI bounce (>60) to EMA20
            elif downtrend and trending and rsi_14[i] > 60 and price < ema20[i]:
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic: reverse signal on opposite condition
        elif position == 1:
            # Exit long if trend breaks or RSI overbought
            if not (uptrend and trending) or rsi_14[i] > 70:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short if trend breaks or RSI oversold
            if not (downtrend and trending) or rsi_14[i] < 30:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA200_Trend_ADX25_RSIPullback_EMA20"
timeframe = "1h"
leverage = 1.0