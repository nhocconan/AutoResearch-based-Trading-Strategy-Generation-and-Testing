#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with weekly ADX trend filter and volume confirmation.
# Long when price breaks above upper Donchian(20) + weekly ADX > 25 + volume > 1.5x average
# Short when price breaks below lower Donchian(20) + weekly ADX > 25 + volume > 1.5x average
# Exit when price crosses back through the Donchian midpoint or volume drops below 0.5x average.
# Designed for low-frequency trading (10-25 trades/year) to minimize fee drag.
# Works in trending markets (ADX > 25) and avoids whipsaws in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period) on daily data
    # Upper = max(high, lookback=20)
    # Lower = min(low, lookback=20)
    # Middle = (upper + lower) / 2
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donch_upper = high_series.rolling(window=20, min_periods=20).max().values
    donch_lower = low_series.rolling(window=20, min_periods=20).min().values
    donch_middle = (donch_upper + donch_lower) / 2
    
    # Calculate ADX (14-period) on weekly data
    # +DM = max(high - previous high, 0) if > previous low - low else 0
    # -DM = max(previous low - low, 0) if > high - previous high else 0
    # TR = max(high - low, high - previous close, previous close - low)
    # +DM smoothed, -DM smoothed, TR smoothed
    # DI+ = 100 * smoothed +DM / smoothed TR
    # DI- = 100 * smoothed -DM / smoothed TR
    # DX = 100 * |DI+ - DI-| / (DI+ + DI-)
    # ADX = smoothed DX
    
    # Calculate True Range
    high_low = high_1w - low_1w
    high_close = np.abs(high_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    low_close = np.abs(low_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    # Calculate Directional Movement
    up_move = high_1w - np.concatenate([[high_1w[0]], high_1w[:-1]])
    down_move = np.concatenate([[low_1w[0]], low_1w[:-1]]) - low_1w
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: smoothed = previous * (1 - 1/period) + current * (1/period)
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Align indicators to lower timeframe (daily)
    donch_upper_aligned = align_htf_to_ltf(prices, df_1d, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1d, donch_lower)
    donch_middle_aligned = align_htf_to_ltf(prices, df_1d, donch_middle)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume filter (20-day average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(donch_upper_aligned[i]) or 
            np.isnan(donch_lower_aligned[i]) or 
            np.isnan(donch_middle_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = donch_upper_aligned[i]
        lower = donch_lower_aligned[i]
        middle = donch_middle_aligned[i]
        adx_val = adx_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_filter = vol > 1.5 * vol_ma
        # Trend filter: ADX > 25 indicates trending market
        trend_filter = adx_val > 25
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian + volume + trend
            if price > upper and vol_filter and trend_filter:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian + volume + trend
            elif price < lower and vol_filter and trend_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through middle or volume drops significantly
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below middle or volume drops below 0.5x average
                if price < middle or vol < 0.5 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above middle or volume drops below 0.5x average
                if price > middle or vol < 0.5 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian_Breakout_WeeklyADX_Volume"
timeframe = "1d"
leverage = 1.0