#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d trend filter (ADX)
# - Long when price breaks above Camarilla H3 AND 4h volume > 1.5x 20-period average AND 1d ADX > 25 (trending market)
# - Short when price breaks below Camarilla L3 AND 4h volume > 1.5x 20-period average AND 1d ADX > 25 (trending market)
# - Exit when price returns to Camarilla pivot point (mean reversion within the day's range)
# - Uses discrete position sizing 0.20 to limit fee churn
# - Camarilla pivots identify intraday support/resistance levels that work well in ranging markets
# - Volume confirms institutional participation in the breakout
# - ADX filter ensures we only trade when there is sufficient trend strength
# - Session filter (08-20 UTC) reduces noise during low-liquidity hours
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)

name = "1h_4h_1d_camarilla_volume_adx_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    # Pre-compute 1h Camarilla Pivots (based on previous day's range)
    # We'll approximate using rolling 24-period high/low (24h = 1 day in 1h timeframe)
    lookback = 24  # 24 periods of 1h = 1 day
    def highest_high(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def lowest_low(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    # Previous day's high/low (24-period lookback)
    prev_day_high = highest_high(high, lookback)
    prev_day_low = lowest_low(low, lookback)
    prev_day_close = np.roll(close, lookback)  # Close from 24 periods ago
    
    # Camarilla levels
    range_val = prev_day_high - prev_day_low
    camarilla_pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
    camarilla_h3 = camarilla_pivot + (range_val * 1.1 / 4)
    camarilla_l3 = camarilla_pivot - (range_val * 1.1 / 4)
    camarilla_h4 = camarilla_pivot + (range_val * 1.1 / 2)
    camarilla_l4 = camarilla_pivot - (range_val * 1.1 / 2)
    
    # Pre-compute 1h ATR (14-period) for stoploss
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[1:15])  # First ATR value
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Pre-compute 4h volume average (20-period)
    volume_4h = df_4h['volume'].values
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    vol_ma_4h = rolling_mean(volume_4h, 20)
    
    # Pre-compute 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d True Range
    tr_1d = np.zeros_like(high_1d)
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr_1d[i] = true_range(high_1d[i], low_1d[i], close_1d[i-1])
    
    # Calculate 1d ATR (14-period)
    atr_1d = np.zeros_like(tr_1d)
    atr_1d[13] = np.mean(tr_1d[1:15])
    for i in range(14, len(tr_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Calculate 1d Directional Movement
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    for i in range(1, len(high_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smooth the DM and TR
    def smoothed_average(arr, window):
        result = np.zeros_like(arr)
        result[window-1] = np.mean(arr[1:window]) if window > 1 else arr[0]
        for i in range(window, len(arr)):
            result[i] = (result[i-1] * (window-1) + arr[i]) / window
        return result
    
    # Only calculate if we have enough data
    if len(tr_1d) >= 14:
        atr_1d_smooth = smoothed_average(tr_1d, 14)
        plus_dm_smooth = smoothed_average(plus_dm, 14)
        minus_dm_smooth = smoothed_average(minus_dm, 14)
        
        # Calculate DI+ and DI-
        plus_di = 100 * plus_dm_smooth / atr_1d_smooth
        minus_di = 100 * minus_dm_smooth / atr_1d_smooth
        
        # Calculate DX and ADX
        dx = np.zeros_like(close_1d)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        # Handle division by zero
        dx = np.where((plus_di + minus_di) == 0, 0, dx)
        
        adx = smoothed_average(dx, 14)
    else:
        adx = np.full_like(close_1d, np.nan)
    
    # Align HTF indicators to 1h timeframe
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid or outside session
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(vol_ma_4h_aligned[i]) or np.isnan(adx_aligned[i]) or
            not in_session[i]):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume confirmation: current 4h volume > 1.5x 20-period average
            # ADX filter: ADX > 25 (trending market)
            volume_ok = volume[i] > 1.5 * vol_ma_4h_aligned[i]
            adx_ok = adx_aligned[i] > 25
            
            # Long conditions: price breaks above Camarilla H3 AND volume AND ADX
            if close[i] > camarilla_h3[i] and volume_ok and adx_ok:
                position = 1
                signals[i] = 0.20
            # Short conditions: price breaks below Camarilla L3 AND volume AND ADX
            elif close[i] < camarilla_l3[i] and volume_ok and adx_ok:
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to Camarilla pivot point
            exit_long = (position == 1 and close[i] <= camarilla_pivot[i])
            exit_short = (position == -1 and close[i] >= camarilla_pivot[i])
            
            # Optional: ATR-based stoploss
            stop_long = (position == 1 and close[i] <= camarilla_h3[i] - 2.0 * atr[i])
            stop_short = (position == -1 and close[i] >= camarilla_l3[i] + 2.0 * atr[i])
            
            if exit_long or exit_short or stop_long or stop_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals

def smoothed_average(arr, window):
    result = np.zeros_like(arr)
    if window <= 1:
        return arr.copy()
    result[window-1] = np.mean(arr[1:window]) if len(arr) >= window else np.nan
    for i in range(window, len(arr)):
        result[i] = (result[i-1] * (window-1) + arr[i]) / window
    return result

def rolling_mean(arr, window):
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        result[i] = np.mean(arr[i - window + 1:i + 1])
    return result

def highest_high(arr, window):
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        result[i] = np.max(arr[i - window + 1:i + 1])
    return result

def lowest_low(arr, window):
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        result[i] = np.min(arr[i - window + 1:i + 1])
    return result

def true_range(h, l, c_prev):
    tr1 = h - l
    tr2 = np.abs(h - c_prev)
    tr3 = np.abs(l - c_prev)
    return np.maximum(tr1, np.maximum(tr2, tr3))