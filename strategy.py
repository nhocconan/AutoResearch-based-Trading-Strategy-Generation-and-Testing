#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h KAMA with 4h trend filter and volume spike for entry timing.
# Uses 4h KAMA trend direction (bullish/bearish) as signal direction.
# Enters on 1h when price crosses KAMA in direction of 4h trend AND volume > 1.5x 20-period average.
# Exits when price crosses back below/above KAMA.
# Includes 4h ADX > 20 to avoid ranging markets.
# Session filter: 08-20 UTC to reduce noise.
# Target: 60-150 total trades over 4 years (15-37/year) with controlled frequency.

name = "1h_KAMA_4hTrend_Volume_Spike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for trend and ADX
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h KAMA for trend direction
    close_4h = df_4h['close'].values
    er_period = 10
    change = np.abs(close_4h - np.roll(close_4h, er_period))
    change[0:er_period] = 0
    volatility = np.abs(np.diff(close_4h, prepend=close_4h[0]))
    volatility = pd.Series(volatility).rolling(window=er_period, min_periods=1).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.66 - 0.06) + 0.06) ** 2
    kama = np.zeros_like(close_4h)
    kama[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
    
    # 4h ADX for trend strength
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_adx = df_4h['close'].values
    
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h_adx, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h_adx, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_4h[0] - low_4h[0]
    
    plus_dm = np.where((high_4h - np.roll(high_4h, 1)) > (np.roll(low_4h, 1) - low_4h), 
                       np.maximum(high_4h - np.roll(high_4h, 1), 0), 0)
    minus_dm = np.where((np.roll(low_4h, 1) - low_4h) > (high_4h - np.roll(high_4h, 1)), 
                        np.maximum(np.roll(low_4h, 1) - low_4h, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx[np.isnan(adx)] = 0
    
    # Align 4h indicators to 1h
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Trend filter: 4h price > KAMA (bullish) or < KAMA (bearish) AND ADX > 20
    trend_bullish = (close_4h > kama) & (adx > 20)
    trend_bearish = (close_4h < kama) & (adx > 20)
    trend_bullish_aligned = align_htf_to_ltf(prices, df_4h, trend_bullish)
    trend_bearish_aligned = align_htf_to_ltf(prices, df_4h, trend_bearish)
    
    # 1h KAMA for entry timing
    change_1h = np.abs(close - np.roll(close, er_period))
    change_1h[0:er_period] = 0
    volatility_1h = np.abs(np.diff(close, prepend=close[0]))
    volatility_1h = pd.Series(volatility_1h).rolling(window=er_period, min_periods=1).sum().values
    er_1h = np.where(volatility_1h != 0, change_1h / volatility_1h, 0)
    sc_1h = (er_1h * (0.66 - 0.06) + 0.06) ** 2
    kama_1h = np.zeros_like(close)
    kama_1h[0] = close[0]
    for i in range(1, len(close)):
        kama_1h[i] = kama_1h[i-1] + sc_1h[i] * (close[i] - kama_1h[i-1])
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(er_period, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_1h[i]) or np.isnan(kama_4h_aligned[i]) or 
            np.isnan(adx_4h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above 1h KAMA, 4h trend bullish, volume spike, session
            long_cond = (close[i] > kama_1h[i]) and (close[i-1] <= kama_1h[i-1]) and \
                        trend_bullish_aligned[i] and volume_spike[i] and session_filter[i]
            # Short: price crosses below 1h KAMA, 4h trend bearish, volume spike, session
            short_cond = (close[i] < kama_1h[i]) and (close[i-1] >= kama_1h[i-1]) and \
                         trend_bearish_aligned[i] and volume_spike[i] and session_filter[i]
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses below 1h KAMA
            if close[i] < kama_1h[i] and close[i-1] >= kama_1h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses above 1h KAMA
            if close[i] > kama_1h[i] and close[i-1] <= kama_1h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals