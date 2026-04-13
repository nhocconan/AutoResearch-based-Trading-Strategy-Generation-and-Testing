#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian channel breakout with 1w ADX trend filter and volume confirmation
    # Long: price breaks above Donchian(20) upper band AND 1w ADX > 25 (trending) AND volume > 1.5x avg
    # Short: price breaks below Donchian(20) lower band AND 1w ADX > 25 (trending) AND volume > 1.5x avg
    # Exit: price crosses Donchian midpoint OR volume dry-up
    # Using 1d timeframe for low trade frequency, Donchian for clear structure,
    # 1w ADX to avoid choppy markets, volume for confirmation.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly ADX(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Align weekly ADX to 1d
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate daily Donchian channels (20-period)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    mid = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        upper[i] = np.max(high[i-lookback+1:i+1])
        lower[i] = np.min(low[i-lookback+1:i+1])
        mid[i] = (upper[i] + lower[i]) / 2
    
    # Get daily volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 = trending market
        trending = adx_1w_aligned[i] > 25
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Breakout conditions
        long_breakout = close[i] > upper[i]
        short_breakout = close[i] < lower[i]
        
        # Exit conditions
        long_exit = close[i] < mid[i]  # Cross below midpoint
        short_exit = close[i] > mid[i]  # Cross above midpoint
        vol_exit = not vol_confirm  # Volume dry-up
        
        if long_breakout and trending and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and trending and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and (long_exit or vol_exit):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (short_exit or vol_exit):
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

name = "1d_1w_donchian_adx_volume_v1"
timeframe = "1d"
leverage = 1.0