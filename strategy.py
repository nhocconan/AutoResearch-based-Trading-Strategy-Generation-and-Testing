#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w ADX trend filter and volume confirmation
    # Long: price > Donchian(20) high AND 1w ADX > 25 (trending) AND volume > 1.5x avg
    # Short: price < Donchian(20) low AND 1w ADX > 25 (trending) AND volume > 1.5x avg
    # Exit: price crosses opposite Donchian band OR volume dry-up
    # Uses 1d timeframe for low trade frequency, Donchian for structure,
    # 1w ADX for regime filter (avoid ranging markets), volume for confirmation.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly ADX(14) for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = -np.diff(low_1w, prepend=low_1w[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Wilder's smoothing (14-period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1w = wilders_smoothing(dx, 14)
    
    # Align weekly ADX to 1d
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate daily Donchian(20)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i >= lookback - 1:
            start_idx = i - lookback + 1
            highest_high[i] = np.max(high[start_idx:i+1])
            lowest_low[i] = np.min(low[start_idx:i+1])
        else:
            highest_high[i] = np.max(high[:i+1]) if i >= 0 else np.nan
            lowest_low[i] = np.min(low[:i+1]) if i >= 0 else np.nan
    
    # Get daily volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 = trending market
        trending = adx_1w_aligned[i] > 25
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Donchian breakout + trending regime + volume confirmation
        long_entry = (close[i] > highest_high[i]) and trending and vol_confirm
        short_entry = (close[i] < lowest_low[i]) and trending and vol_confirm
        
        # Exit logic: opposite Donchian break OR volume dry-up
        long_exit = (close[i] < lowest_low[i]) or not vol_confirm
        short_exit = (close[i] > highest_high[i]) or not vol_confirm
        
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

name = "1d_1w_donchian_adx_volume_v1"
timeframe = "1d"
leverage = 1.0