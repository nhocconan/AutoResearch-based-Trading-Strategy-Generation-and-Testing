#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above Donchian upper band AND 1d ADX > 25 AND volume > 1.5x 20-period average
# Short when price breaks below Donchian lower band AND 1d ADX > 25 AND volume > 1.5x 20-period average
# ATR-based trailing stop (2.5x ATR) for risk management
# Discrete position sizing (0.25) to minimize fee drag
# Target: 12-25 trades/year on 12h timeframe (~50-100 total over 4 years)
# Works in bull markets via long breakouts with strong 1d trend (ADX>25)
# Works in bear markets via short breakdowns with strong 1d trend (ADX>25)
# Uses 1d timeframe for HTF trend filter to capture major market regimes

name = "12h_Donchian_20_Breakout_1dADX_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend filter (using 14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_first = np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])
    tr_1d = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus_first = np.maximum(high_1d[0] - high_1d[0], 0)  # Always 0 for first bar
    dm_minus_first = np.maximum(low_1d[0] - low_1d[0], 0)   # Always 0 for first bar
    dm_plus_1d = np.concatenate([[dm_plus_first], dm_plus])
    dm_minus_1d = np.concatenate([[dm_minus_first], dm_minus])
    
    # Smoothed TR, DM+ and DM- using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(values) >= period:
            result[period-1] = np.mean(values[:period])
            # Wilder's smoothing: today = (1-alpha)*yesterday + alpha*today
            for i in range(period, len(values)):
                result[i] = (1-alpha) * result[i-1] + alpha * values[i]
        return result
    
    atr_1d = wilders_smoothing(tr_1d, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus_1d, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus_1d, 14)
    
    # DI+ and DI-
    di_plus_1d = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus_1d = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx_1d = np.where((di_plus_1d + di_minus_1d) != 0, 
                     100 * np.abs(di_plus_1d - di_minus_1d) / (di_plus_1d + di_minus_1d), 0)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Align 1d ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Donchian channels (20-period) on 12h data
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_upper = rolling_max(high, 20)
    donchian_lower = rolling_min(low, 20)
    
    # Calculate ATR for stoploss (using 14-period on 12h)
    tr1_12h = high[1:] - low[1:]
    tr2_12h = np.abs(high[1:] - close[:-1])
    tr3_12h = np.abs(low[1:] - close[:-1])
    tr_first_12h = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr_12h = np.concatenate([[tr_first_12h], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume spike confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 20:
            vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = max(100, 50, 50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_adx_1d = adx_1d_aligned[i]
        curr_atr = atr_12h[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Skip if indicators are not available
        if (np.isnan(curr_adx_1d) or np.isnan(curr_atr) or 
            np.isnan(curr_upper) or np.isnan(curr_lower) or
            np.isnan(curr_vol_ma)):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * curr_vol_ma if curr_vol_ma > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.5 * ATR below highest high
            stop_price = highest_high_since_entry - 2.5 * curr_atr
            # Exit conditions: price below trailing stop
            if curr_close < stop_price:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # Trailing stop: 2.5 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.5 * curr_atr
            # Exit conditions: price above trailing stop
            if curr_close > stop_price:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper band AND 1d ADX > 25 AND volume spike
            if curr_close > curr_upper and curr_adx_1d > 25 and vol_spike:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: price breaks below Donchian lower band AND 1d ADX > 25 AND volume spike
            elif curr_close < curr_lower and curr_adx_1d > 25 and vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals