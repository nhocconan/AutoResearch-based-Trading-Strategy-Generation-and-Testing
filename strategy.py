#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Daily Donchian Breakout with Volume and ADX Filter
# Hypothesis: Daily Donchian(20) breakouts on 4h, confirmed by volume and ADX>25,
# capture institutional moves across bull/bear markets. Uses daily trend filter
# to avoid counter-trend trades. Target: 20-40 trades/year (80-160 total).
# Volume filter reduces false breakouts. ADX ensures trending conditions.

name = "4h_daily_donchian_breakout_volume_adx_v6"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian and trend
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Daily Donchian(20)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    donch_high = rolling_max(daily_high, 20)
    donch_low = rolling_min(daily_low, 20)
    
    # Align to 4h
    donch_high_4h = align_htf_to_ltf(prices, df_daily, donch_high)
    donch_low_4h = align_htf_to_ltf(prices, df_daily, donch_low)
    
    # Daily EMA(50) for trend filter
    daily_close = df_daily['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_daily, ema_50)
    
    # ADX(14) on daily
    def calculate_adx(high, low, close, period=14):
        # TR
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = 0
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # +DM, -DM
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth TR, +DM, -DM
        def wilder_smooth(data, period):
            smoothed = np.full_like(data, np.nan, dtype=float)
            if len(data) < period:
                return smoothed
            smoothed[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                smoothed[i] = (smoothed[i-1] * (period-1) + data[i]) / period
            return smoothed
        
        atr = wilder_smooth(tr, period)
        plus_di = 100 * wilder_smooth(plus_dm, period) / np.where(atr != 0, atr, 1e-10)
        minus_di = 100 * wilder_smooth(minus_dm, period) / np.where(atr != 0, atr, 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), 1e-10)
        adx = wilder_smooth(dx, period)
        return adx
    
    adx = calculate_adx(daily_high, daily_low, daily_close, 14)
    adx_4h = align_htf_to_ltf(prices, df_daily, adx)
    
    # Volume MA(20) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if required data not available
        if (np.isnan(donch_high_4h[i]) or np.isnan(donch_low_4h[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(adx_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: above average
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price touches lower Donchian or trend changes
            if close[i] <= donch_low_4h[i] or close[i] < ema_50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price touches upper Donchian or trend changes
            if close[i] >= donch_high_4h[i] or close[i] > ema_50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume and ADX filters
            if vol_ok and adx_4h[i] > 25:
                # Long breakout in uptrend
                if close[i] > donch_high_4h[i] and close[i] > ema_50_4h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown in downtrend
                elif close[i] < donch_low_4h[i] and close[i] < ema_50_4h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals