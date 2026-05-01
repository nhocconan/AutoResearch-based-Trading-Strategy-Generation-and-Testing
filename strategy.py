#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ADX trend filter + volume confirmation.
# Long when price breaks above 20-bar Donchian high with 1d ADX > 25 and volume > 1.5x 20-bar average.
# Short when price breaks below 20-bar Donchian low with 1d ADX > 25 and volume > 1.5x 20-bar average.
# Uses ATR-based trailing stop: exit long when price < highest_high_since_entry - 2.5*ATR(14),
# exit short when price > lowest_low_since_entry + 2.5*ATR(14).
# Discrete sizing 0.30 to balance return and drawdown. Designed for 4h timeframe to capture medium-term trends.
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) via 1d ADX filter.

name = "4h_Donchian20_1dADX_VolumeConfirm_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ADX calculation (14-period)
    def calculate_adx(high_arr, low_arr, close_arr, period=14):
        # True Range
        tr1 = np.abs(high_arr[1:] - low_arr[1:])
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align length
        
        # Directional Movement
        up_move = high_arr[1:] - high_arr[:-1]
        down_move = low_arr[:-1] - low_arr[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
        def wilders_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        atr = wilders_smoothing(tr, period)
        plus_dm_smooth = wilders_smoothing(plus_dm, period)
        minus_dm_smooth = wilders_smoothing(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilders_smoothing(dx, period)
        return adx
    
    # Calculate 1d ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high_values = donchian_high.values
    donchian_low_values = donchian_low.values
    
    # 4h ATR calculation (14-period) for trailing stop
    def calculate_atr(high_arr, low_arr, close_arr, period=14):
        # True Range
        tr1 = np.abs(high_arr[1:] - low_arr[1:])
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align length
        
        # Wilder's smoothing
        def wilders_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        atr = wilders_smoothing(tr, period)
        return atr
    
    atr_4h = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 50  # warmup for Donchian, ADX, and ATR
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 4h timeframe
        hour = hours[i]
        
        if np.isnan(adx_1d_aligned[i]) or np.isnan(donchian_high_values[i]) or np.isnan(donchian_low_values[i]) or np.isnan(atr_4h[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_adx = adx_1d_aligned[i]
        curr_atr = atr_4h[i]
        curr_donchian_high = donchian_high_values[i]
        curr_donchian_low = donchian_low_values[i]
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        if i < 20 + start_idx:
            signals[i] = 0.0
            continue
            
        vol_ma = np.mean(volume[i-20:i])  # 20-period simple moving average
        if vol_ma <= 0:
            signals[i] = 0.0
            continue
        volume_confirm = curr_vol > (vol_ma * 1.5)
        
        # Donchian breakout signals
        bullish_breakout = curr_close > curr_donchian_high
        bearish_breakout = curr_close < curr_donchian_low
        
        if position == 0:  # Flat - look for new entries
            # Long: bullish Donchian breakout AND ADX > 25 AND volume confirmation
            if (bullish_breakout and 
                curr_adx > 25 and 
                volume_confirm):
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            # Short: bearish Donchian breakout AND ADX > 25 AND volume confirmation
            elif (bearish_breakout and 
                  curr_adx > 25 and 
                  volume_confirm):
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            
            # Exit conditions:
            # 1. ATR trailing stop: price < highest_high_since_entry - 2.5*ATR
            # 2. Trend weakening: ADX < 20
            # 3. Opposite Donchian breakout
            if (curr_close < highest_since_entry - 2.5 * curr_atr or
                curr_adx < 20 or
                bearish_breakout):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            
            # Exit conditions:
            # 1. ATR trailing stop: price > lowest_low_since_entry + 2.5*ATR
            # 2. Trend weakening: ADX < 20
            # 3. Opposite Donchian breakout
            if (curr_close > lowest_since_entry + 2.5 * curr_atr or
                curr_adx < 20 or
                bullish_breakout):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals