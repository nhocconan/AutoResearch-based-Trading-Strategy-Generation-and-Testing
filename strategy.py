#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + 12h ADX regime filter
    # Long: Bull Power > 0 (close > EMA13) + Bear Power < 0 (low < EMA13) + 12h ADX > 25 (trending)
    # Short: Bear Power < 0 (low < EMA13) + Bull Power < 0 (close < EMA13) + 12h ADX > 25 (trending)
    # Uses discrete sizing (0.25) to minimize fee drag
    # Target: 12-37 trades/year to stay within 6h optimal range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ADX (14-period)
    # True Range
    tr_12h = np.zeros(len(close_12h))
    for i in range(1, len(close_12h)):
        tr_12h[i] = max(
            high_12h[i] - low_12h[i],
            abs(high_12h[i] - close_12h[i-1]),
            abs(low_12h[i] - close_12h[i-1])
        )
    tr_12h[0] = high_12h[0] - low_12h[0]
    
    # Directional Movement
    dm_plus_12h = np.zeros(len(close_12h))
    dm_minus_12h = np.zeros(len(close_12h))
    for i in range(1, len(close_12h)):
        up_move = high_12h[i] - high_12h[i-1]
        down_move = low_12h[i-1] - low_12h[i]
        dm_plus_12h[i] = up_move if up_move > down_move and up_move > 0 else 0
        dm_minus_12h[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smoothed TR, DM+ , DM- (Wilder's smoothing = alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    tr_14_12h = np.zeros(len(close_12h))
    dm_plus_14_12h = np.zeros(len(close_12h))
    dm_minus_14_12h = np.zeros(len(close_12h))
    
    # Initial values (simple average)
    tr_14_12h[period-1] = np.mean(tr_12h[:period])
    dm_plus_14_12h[period-1] = np.mean(dm_plus_12h[:period])
    dm_minus_14_12h[period-1] = np.mean(dm_minus_12h[:period])
    
    # Wilder's smoothing
    for i in range(period, len(close_12h)):
        tr_14_12h[i] = tr_14_12h[i-1] * (1 - alpha) + alpha * tr_12h[i]
        dm_plus_14_12h[i] = dm_plus_14_12h[i-1] * (1 - alpha) + alpha * dm_plus_12h[i]
        dm_minus_14_12h[i] = dm_minus_14_12h[i-1] * (1 - alpha) + alpha * dm_minus_12h[i]
    
    # Directional Indicators
    di_plus_12h = 100 * dm_plus_14_12h / tr_14_12h
    di_minus_12h = 100 * dm_minus_14_12h / tr_14_12h
    
    # DX and ADX
    dx_12h = np.zeros(len(close_12h))
    for i in range(len(close_12h)):
        if di_plus_12h[i] + di_minus_12h[i] > 0:
            dx_12h[i] = 100 * abs(di_plus_12h[i] - di_minus_12h[i]) / (di_plus_12h[i] + di_minus_12h[i])
        else:
            dx_12h[i] = 0
    
    # ADX (smoothed DX)
    adx_12h = np.zeros(len(close_12h))
    adx_12h[2*period-1] = np.mean(dx_12h[period:2*period])
    for i in range(2*period, len(close_12h)):
        adx_12h[i] = adx_12h[i-1] * (1 - alpha) + alpha * dx_12h[i]
    
    # Align ADX to 6h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = close - ema_13_6h  # Bull Power = Close - EMA13
    bear_power = low - ema_13_6h    # Bear Power = Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    # Calculate ATR using true range approximation for 6h timeframe
    atr_6h = np.zeros(n)
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        if i < 14:
            atr_6h[i] = tr  # Simple average for warmup
        else:
            atr_6h[i] = 0.93 * atr_6h[i-1] + 0.07 * tr  # Wilder's smoothing
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_12h_aligned[i]) or 
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: 12h ADX > 25 (trending market)
        trending = adx_12h_aligned[i] > 25
        
        # Elder Ray conditions
        # Long: Bull Power > 0 AND Bear Power < 0 (market in bullish phase but not overextended)
        # Short: Bear Power < 0 AND Bull Power < 0 (market in bearish phase but not overextended)
        long_condition = (bull_power[i] > 0) and (bear_power[i] < 0) and trending
        short_condition = (bear_power[i] < 0) and (bull_power[i] < 0) and trending
        
        # Stoploss: 2x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.0 * atr_6h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.0 * atr_6h[i]
        
        # Execute signals
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "6h_12h_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0