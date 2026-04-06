#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour momentum strategy using 4-hour ADX for trend strength and 1-day RSI for overbought/oversold conditions.
# Uses 4h ADX > 25 to filter trending markets and 1d RSI < 30 or > 70 for mean-reversion entries.
# Trades only during 08-20 UTC session to reduce noise.
# Targets 15-30 trades/year (60-120 over 4 years) to minimize fee drag.
# Works in bull/bear by combining trend filter with mean-reversion entries.

name = "1h_adx25_rsi_meanrev_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # 4h ADX(14) for trend strength
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate directional movement
    up_move = np.diff(high_4h)
    down_move = -np.diff(low_4h)  # negative of diff to get positive values
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # True range
    tr_4h = np.maximum(
        high_4h[1:] - low_4h[1:],
        np.abs(high_4h[1:] - close_4h[:-1]),
        np.abs(low_4h[1:] - close_4h[:-1])
    )
    
    # Smoothing (Wilder's smoothing)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate smoothed values
    if len(tr_4h) >= 14:
        atr_4h = wilder_smooth(tr_4h, 14)
        plus_dm_smooth = wilder_smooth(plus_dm, 14)
        minus_dm_smooth = wilder_smooth(minus_dm, 14)
        
        # DI values
        plus_di_4h = np.full_like(atr_4h, np.nan)
        minus_di_4h = np.full_like(atr_4h, np.nan)
        mask = ~np.isnan(atr_4h) & (atr_4h != 0)
        plus_di_4h[mask] = 100 * plus_dm_smooth[mask] / atr_4h[mask]
        minus_di_4h[mask] = 100 * minus_dm_smooth[mask] / atr_4h[mask]
        
        # DX and ADX
        dx_4h = np.full_like(atr_4h, np.nan)
        mask_dx = (~np.isnan(plus_di_4h) & ~np.isnan(minus_di_4h) & 
                  ((plus_di_4h + minus_di_4h) != 0))
        dx_4h[mask_dx] = 100 * np.abs(plus_di_4h[mask_dx] - minus_di_4h[mask_dx]) / \
                         (plus_di_4h[mask_dx] + minus_di_4h[mask_dx])
        
        adx_4h = wilder_smooth(dx_4h, 14)
    else:
        adx_4h = np.full(len(close_4h), np.nan)
    
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # 1d RSI(14) for overbought/oversold
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    def rsi(prices, period=14):
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)
        delta = np.diff(prices)
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        up_smoothed = np.full_like(up, np.nan)
        down_smoothed = np.full_like(down, np.nan)
        up_smoothed[period-1] = np.mean(up[:period])
        down_smoothed[period-1] = np.mean(down[:period])
        for i in range(period, len(up)):
            up_smoothed[i] = (up_smoothed[i-1] * (period-1) + up[i]) / period
            down_smoothed[i] = (down_smoothed[i-1] * (period-1) + down[i]) / period
        
        rs = np.where(down_smoothed != 0, up_smoothed / down_smoothed, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_1d = rsi(close_1d, 14)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(30, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: RSI returns to neutral or stoploss hit
            if (rsi_aligned[i] >= 50 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI returns to neutral or stoploss hit
            if (rsi_aligned[i] <= 50 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries
            # Only trade when 4h ADX indicates trending market
            if adx_aligned[i] > 25:
                # Long: RSI oversold (<30) in uptrend
                if rsi_aligned[i] < 30:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short: RSI overbought (>70) in uptrend
                elif rsi_aligned[i] > 70:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals