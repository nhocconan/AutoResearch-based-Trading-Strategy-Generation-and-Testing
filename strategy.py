#!/usr/bin/env python3
"""
1h Volume-Weighted RSI with 4h Trend and 1d Regime Filter
Hypothesis: In strong trends (4h), RSI pullbacks offer high-probability entries; 
1d regime filter (ADX) avoids ranging markets. Volume confirms momentum.
Targets 15-35 trades/year by requiring confluence of trend, momentum, and volume.
Works in bull (buy pullbacks) and bear (sell rallies) via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_vwrsi_4h_trend_1d_adx_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI (14) - Wilder's smoothing
    def rsi_wilder(close, period=14):
        delta = np.diff(close, prepend=close[0])
        up = np.where(delta > 0, delta, 0.0)
        down = np.where(delta < 0, -delta, 0.0)
        # Wilder's smoothing: alpha = 1/period
        up_ewm = np.zeros_like(close)
        down_ewm = np.zeros_like(close)
        up_ewm[0] = up[0]
        down_ewm[0] = down[0]
        for i in range(1, len(close)):
            up_ewm[i] = (up[i] + up_ewm[i-1] * (period-1)) / period
            down_ewm[i] = (down[i] + down_ewm[i-1] * (period-1)) / period
        rs = np.where(down_ewm != 0, up_ewm / down_ewm, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = rsi_wilder(close, 14)
    
    # Volume-weighted RSI: weight RSI by volume ratio
    vol_ratio = volume / (np.where(np.convolve(volume, np.ones(20), 'same')/20, np.convolve(volume, np.ones(20), 'same')/20, 1))
    vwrsi = np.where(vol_ratio > 0, rsi * vol_ratio, rsi)
    # Normalize VWRSI to 0-100 range for consistency
    vwrsi = np.clip(vwrsi, 0, 100)
    
    # 4h trend filter: EMA21
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 21:
        ema_4h[20] = np.mean(close_4h[:21])
        for i in range(21, len(close_4h)):
            ema_4h[i] = (close_4h[i] * 2 + ema_4h[i-1] * 19) / 21
    trend_4h = np.where(close_4h > ema_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # 1d regime filter: ADX(14) > 25 for trending market
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    # Directional Movement
    up_move = np.concatenate([[0], np.diff(high_1d)])
    down_move = np.concatenate([[0], -np.diff(low_1d)])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilders_smooth(arr, period):
        smoothed = np.full_like(arr, np.nan)
        if len(arr) >= period:
            smoothed[period-1] = np.nansum(arr[:period])
            for i in range(period, len(arr)):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    atr_1d = wilders_smooth(tr, 14)
    plus_di_1d = 100 * wilders_smooth(plus_dm, 14) / np.where(atr_1d != 0, atr_1d, 1)
    minus_di_1d = 100 * wilders_smooth(minus_dm, 14) / np.where(atr_1d != 0, atr_1d, 1)
    dx = 100 * np.abs(plus_di_1d - minus_di_1d) / np.where((plus_di_1d + minus_di_1d) != 0, (plus_di_1d + minus_di_1d), 1)
    adx_1d = wilders_smooth(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_filter = volume > vol_ma * 1.3
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start = max(100, 21)
    
    for i in range(start, n):
        if np.isnan(vwrsi[i]) or np.isnan(trend_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long
            # Exit: VWRSI > 70 (overbought) OR trend turns down OR ADX < 20 (losing trend)
            if (vwrsi[i] > 70 or
                trend_4h_aligned[i] == -1 or
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short
            # Exit: VWRSI < 30 (oversold) OR trend turns up OR ADX < 20
            if (vwrsi[i] < 30 or
                trend_4h_aligned[i] == 1 or
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Entry conditions
            # Long: VWRSI < 30 (oversold) in uptrend with strong trend and volume
            if (vwrsi[i] < 30 and
                trend_4h_aligned[i] == 1 and
                adx_1d_aligned[i] > 25 and
                volume_filter[i] and
                session_filter[i]):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: VWRSI > 70 (overbought) in downtrend with strong trend and volume
            elif (vwrsi[i] > 70 and
                  trend_4h_aligned[i] == -1 and
                  adx_1d_aligned[i] > 25 and
                  volume_filter[i] and
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals