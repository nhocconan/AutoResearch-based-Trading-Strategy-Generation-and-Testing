#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h ADX + Volume Breakout with 12h trend filter
    # Long: ADX(14) > 25 (trending) AND +DI > -DI (bullish momentum) AND volume > 1.5x 20-period average AND price > 12h EMA20
    # Short: ADX(14) > 25 AND -DI > +DI (bearish momentum) AND volume > 1.5x 20-period average AND price < 12h EMA20
    # Exit: ADX < 20 (trend weakening) OR price crosses 12h EMA20 in opposite direction
    # Using 12h for EMA20 trend filter (structure) and ADX for trend strength, 6h only for entry timing
    # Discrete position sizing (0.25) to balance return and drawdown
    # Target: 12-37 trades/year (~50-150 over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX and EMA20 (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ADX components
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(values)):
            result[i] = result[i-1] * (1 - 1/period) + values[i] * (1/period)
        return result
    
    period = 14
    tr_smooth = wilders_smoothing(tr, period)
    plus_dm_smooth = wilders_smoothing(plus_dm, period)
    minus_dm_smooth = wilders_smoothing(minus_dm, period)
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    # Align 12h ADX, +DI, -DI to 6h (wait for completed 12h bar)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    plus_di_12h_aligned = align_htf_to_ltf(prices, df_12h, plus_di)
    minus_di_12h_aligned = align_htf_to_ltf(prices, df_12h, minus_di)
    
    # 12h EMA20 for trend filter
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: >1.5x 20-period average (to reduce false signals)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(plus_di_12h_aligned[i]) or 
            np.isnan(minus_di_12h_aligned[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trend filter: only long if price > 12h EMA20, only short if price < 12h EMA20
        long_trend_ok = close[i] > ema_12h_aligned[i]
        short_trend_ok = close[i] < ema_12h_aligned[i]
        
        # ADX trend strength and direction
        strong_trend = adx_12h_aligned[i] > 25
        weak_trend = adx_12h_aligned[i] < 20  # exit condition
        bullish_momentum = plus_di_12h_aligned[i] > minus_di_12h_aligned[i]
        bearish_momentum = minus_di_12h_aligned[i] > plus_di_12h_aligned[i]
        
        # Entry logic: ADX breakout + volume + trend
        long_entry = strong_trend and bullish_momentum and vol_confirm and long_trend_ok
        short_entry = strong_trend and bearish_momentum and vol_confirm and short_trend_ok
        
        # Exit logic: trend weakening OR price crosses EMA in opposite direction
        long_exit = weak_trend or (position == 1 and close[i] < ema_12h_aligned[i])
        short_exit = weak_trend or (position == -1 and close[i] > ema_12h_aligned[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_adx_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0