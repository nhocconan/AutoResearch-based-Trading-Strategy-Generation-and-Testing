#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d volume spike (1.5x) and ADX > 25
    # Uses tighter volume filter and moderate ADX to reduce trades while maintaining edge
    # Long when price > upper Donchian + volume spike + ADX > 25
    # Short when price < lower Donchian + volume spike + ADX > 25
    # Exit when price crosses middle Donchian OR ADX < 20
    # Discrete position sizing: 0.25
    # Target: 75-150 total trades over 4 years (~19-37/year)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Upper channel: highest high of last 20 days
    upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low of last 20 days
    lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    # Middle channel: average of upper and lower
    middle = (upper + lower) / 2
    
    # Align 1d Donchian levels to 4h
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    middle_aligned = align_htf_to_ltf(prices, df_1d, middle)
    
    # Calculate 1d volume average (20-period) with min_periods
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1d ADX (14-period) for trend filter
    # TR = max(|high-low|, |high-prev_close|, |low-prev_close|)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(np.roll(high_1d, 1) - close_1d)
    tr3 = np.abs(np.roll(low_1d, 1) - close_1d)
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +DM = high - prev_high (if positive and > prev_low - low)
    up_move = np.roll(high_1d, 1) - high_1d
    down_move = low_1d - np.roll(low_1d, 1)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = np.nan
    minus_dm[0] = np.nan
    
    # Wilder's smoothing function
    def wilders_smoothing(data, period):
        """Wilder's smoothing (equivalent to EMA with alpha=1/period)"""
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.empty_like(data)
        result[:] = np.nan
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Rest is Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_smooth = wilders_smoothing(tr, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX = |DI+ - DI-| / (DI+ + DI-) * 100
    dx = np.where((plus_di + minus_di) != 0, 
                  np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 
                  0)
    
    # ADX = Wilder's smoothing of DX
    adx = wilders_smoothing(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(middle_aligned[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5 * 20-period average
        vol_1d_current = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_current)
        volume_confirm = vol_1d_aligned[i] > 1.5 * vol_ma_aligned[i]
        
        # ADX trend filter: strong trend (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        
        # Breakout conditions
        bullish_breakout = close[i] > upper_aligned[i] and volume_confirm and strong_trend
        bearish_breakout = close[i] < lower_aligned[i] and volume_confirm and strong_trend
        
        # Exit conditions: price returns to middle Donchian OR trend weakens (ADX < 20)
        trend_weakening = adx_aligned[i] < 20
        long_exit = close[i] < middle_aligned[i] or trend_weakening or bearish_breakout
        short_exit = close[i] > middle_aligned[i] or trend_weakening or bullish_breakout
        
        if bullish_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_breakout and position != -1:
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

name = "4h_1d_donchian_breakout_volume_adx_v7"
timeframe = "4h"
leverage = 1.0