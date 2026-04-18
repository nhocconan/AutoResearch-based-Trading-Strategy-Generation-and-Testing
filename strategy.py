#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R1/S1 breakout with daily volume confirmation and ADX trend filter.
# Camarilla pivot levels provide high-probability reversal/breakout levels based on prior day's price action.
# Daily volume confirmation ensures institutional participation in breakouts.
# ADX filter (from 1d) ensures we only trade in trending regimes to avoid chop.
# Designed for low trade frequency (15-30/year) to minimize fee drag in 12h timeframe.
# Works in bull markets (breakouts above R1) and bear markets (breakouts below S1).
name = "12h_Camarilla_R1_S1_Breakout_Volume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots, volume MA, and ADX (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for R1 and S1 using previous day's data
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    high_d_prev = np.concatenate([[np.nan], high_d[:-1]])
    low_d_prev = np.concatenate([[np.nan], low_d[:-1]])
    close_d_prev = np.concatenate([[np.nan], close_d[:-1]])
    
    rang = high_d_prev - low_d_prev
    r1 = close_d_prev + rang * 1.1 / 12
    s1 = close_d_prev - rang * 1.1 / 12
    
    # Calculate daily ADX (14-period) for trend strength
    # ADX calculation requires +DI and -DI
    # True Range
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # +DM and -DM
    up_move = high_d[1:] - high_d[:-1]
    down_move = low_d[:-1] - low_d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/14)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
                else:
                    result[i] = np.nan
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Align daily indicators to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 20-period average volume for confirmation (daily)
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current daily volume above average
        # We use the aligned daily volume MA, so we need to get the current day's volume
        # Find the index in df_1d that corresponds to current 12h bar
        vol_confirm = False
        if i < len(vol_ma_20_aligned) and not np.isnan(vol_ma_20_aligned[i]):
            # Get the current day's volume from the aligned data
            # Since we're using 12h timeframe, we check if current volume exceeds daily average
            # We approximate by comparing current 12h volume to a fraction of daily average
            # More direct: use the fact that vol_ma_20_aligned is already the daily average aligned
            # We'll use a simplified check: current 12h volume > 0 (always true when volume exists)
            # Better: use volume spike detection - current volume > 1.5 * average 12h volume
            # For simplicity and to avoid look-ahead, we'll use daily volume MA as threshold
            # and check if the 12h bar's volume is significant relative to daily context
            # Since we don't have intraday volume breakdown, we'll use a simpler approach:
            # Volume confirmation based on whether we're in a high-volume day
            # We'll use the actual daily volume (not just MA) for confirmation
            pass  # We'll handle volume confirmation differently below
        
        # Instead, let's use a more direct approach: check if current 12h bar volume is elevated
        # Calculate 20-period average of 12h volume for confirmation
        if i >= 20:
            vol_ma_20_12h = np.mean(volume[i-20:i])
            vol_confirm = volume[i] > vol_ma_20_12h
        else:
            vol_confirm = False
        
        # ADX filter: only trade when ADX > 25 (trending market)
        adx_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above R1 AND volume confirmation AND ADX filter
            long_breakout = close[i] > r1_aligned[i]
            if vol_confirm and adx_filter and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume confirmation AND ADX filter
            elif vol_confirm and adx_filter and close[i] < s1_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below S1 OR ADX drops below 20 (trend weakening)
            exit_condition = close[i] < s1_aligned[i] or adx_aligned[i] < 20
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R1 OR ADX drops below 20 (trend weakening)
            exit_condition = close[i] > r1_aligned[i] or adx_aligned[i] < 20
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals