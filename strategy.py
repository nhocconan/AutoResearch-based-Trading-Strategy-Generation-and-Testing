#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with volume confirmation and ADX filter.
# Camarilla pivot levels (R1/S1) provide high-probability support/resistance levels derived from prior day's range.
# Breakouts above R1 or below S1 with volume confirmation indicate institutional participation.
# ADX filter ensures we only trade in trending conditions (ADX > 25) to avoid false breakouts in ranging markets.
# Designed for low trade frequency (12-37/year) to minimize fee drag in 12h timeframe.
# Works in bull markets (breakouts above R1) and bear markets (breakouts below S1).
name = "12h_Camarilla_R1S1_Breakout_Volume_ADX"
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
    
    # Get daily data for Camarilla pivots and ADX (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day's range
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C = (H+L+CLOSE)/3 (typical price)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    daily_range = df_1d['high'] - df_1d['low']
    r1 = typical_price + daily_range * 1.1 / 12
    s1 = typical_price - daily_range * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # Calculate ADX (14-period) on daily timeframe
    # +DM, -DM, TR calculation
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Directional Movement
    up_move = high_d[1:] - high_d[:-1]
    down_move = low_d[:-1] - low_d[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # True Range
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Smoothing (Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
                else:
                    result[i] = np.nan
        return result
    
    period = 14
    tr_smooth = WilderSmooth(tr, period)
    plus_dm_smooth = WilderSmooth(plus_dm, period)
    minus_dm_smooth = WilderSmooth(minus_dm, period)
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = WilderSmooth(dx, period)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 24-period average volume for confirmation (2 days of 12h data)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_24[i]
        
        # ADX filter: trending condition (ADX > 25)
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