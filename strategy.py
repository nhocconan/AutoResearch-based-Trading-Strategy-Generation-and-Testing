#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above 4h Donchian upper (20) AND 1d ADX > 25 AND 1h volume > 1.5x 20-bar average.
# Short when price breaks below 4h Donchian lower (20) AND 1d ADX > 25 AND 1h volume > 1.5x 20-bar average.
# Exit when price crosses 4h Donchian midline (10-bar average of upper/lower) OR ADX < 20 (trend weak).
# Uses discrete position size 0.20. Donchian provides clear structure, ADX filters choppy markets, volume confirms breakout strength.
# 1h timeframe targets 60-150 total trades over 4 years (15-37/year) by using 4h for direction and 1h only for precise timing.
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 4h data once before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Get 1d data once before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 4h Indicators: Donchian Channel (20) ===
    # Upper = max(high, 20), Lower = min(low, 20), Middle = (upper + lower) / 2
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.max(arr[i - window + 1:i + 1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.min(arr[i - window + 1:i + 1])
        return res
    
    donch_20_upper_4h = rolling_max(high_4h, 20)
    donch_20_lower_4h = rolling_min(low_4h, 20)
    donch_20_middle_4h = (donch_20_upper_4h + donch_20_lower_4h) / 2.0
    
    # === 1d Indicators: ADX (14) ===
    # ADX calculation using Wilder's smoothing
    def calculate_adx(high, low, close, window=14):
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # prepend NaN for index alignment
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/window)
        def wilder_smoothing(arr, window):
            res = np.full_like(arr, np.nan)
            alpha = 1.0 / window
            # First value: simple average
            if window < len(arr):
                res[window-1] = np.nanmean(arr[:window])
                for i in range(window, len(arr)):
                    res[i] = alpha * arr[i] + (1 - alpha) * res[i-1]
            return res
        
        atr = wilder_smoothing(tr, window)
        plus_dm_smooth = wilder_smoothing(plus_dm, window)
        minus_dm_smooth = wilder_smoothing(minus_dm, window)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = wilder_smoothing(dx, window)
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align all indicators to primary timeframe (1h)
    donch_20_upper_aligned = align_htf_to_ltf(prices, df_4h, donch_20_upper_4h)
    donch_20_lower_aligned = align_htf_to_ltf(prices, df_4h, donch_20_lower_4h)
    donch_20_middle_aligned = align_htf_to_ltf(prices, df_4h, donch_20_middle_4h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Pre-compute volume filter: 1h volume > 1.5x 20-bar average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50  # Donchian(20) + ADX(14) + volume MA(20) need sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_20_upper_aligned[i]) or np.isnan(donch_20_lower_aligned[i]) or 
            np.isnan(donch_20_middle_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Session filter
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        price = close[i]
        vol_ok = volume_filter[i]
        upper = donch_20_upper_aligned[i]
        lower = donch_20_lower_aligned[i]
        middle = donch_20_middle_aligned[i]
        adx = adx_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < Donchian middle OR ADX < 20 (trend weak)
            if (price < middle) or (adx < 20):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > Donchian middle OR ADX < 20 (trend weak)
            if (price > middle) or (adx < 20):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0 and vol_ok:
            # LONG: price > Donchian upper AND ADX > 25 (strong uptrend)
            if (price > upper) and (adx > 25):
                signals[i] = 0.20
                position = 1
            
            # SHORT: price < Donchian lower AND ADX > 25 (strong downtrend)
            elif (price < lower) and (adx > 25):
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20  # maintain position
    
    return signals

name = "1h_4hDonchian20_1dADX25_VolumeFilter_Session_V1"
timeframe = "1h"
leverage = 1.0