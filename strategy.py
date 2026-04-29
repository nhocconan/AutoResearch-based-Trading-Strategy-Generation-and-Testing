#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h ADX(14) trend filter and volume confirmation
# Long when price breaks above Donchian upper band AND 12h ADX > 25 AND volume > 1.5x 20-bar avg
# Short when price breaks below Donchian lower band AND 12h ADX > 25 AND volume > 1.5x 20-bar avg
# Exit when price retraces to Donchian midpoint (mean reversion within the channel)
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 12-30 trades/year on 6h.
# Works in bull markets by capturing breakouts with trend alignment, and in bear markets
# by shorting breakdowns with trend filter preventing whipsaws. Volume confirmation
# ensures breakouts have conviction. Donchian midpoint exit provides symmetrical risk/reward.

name = "6h_Donchian20_12hADX25_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ADX(14) for trend filter
    # ADX calculation: +DM, -DM, TR, then smoothed, then DX, then ADX
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # prepend NaN for alignment
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed values using Wilder's smoothing (alpha = 1/period)
        def wilder_smooth(values, period):
            smoothed = np.full_like(values, np.nan)
            if len(values) < period:
                return smoothed
            # First value is simple average
            smoothed[period-1] = np.nanmean(values[:period])
            # Subsequent values: smoothed[i] = smoothed[i-1] - (smoothed[i-1]/period) + values[i]/period
            for i in range(period, len(values)):
                if not np.isnan(values[i]):
                    smoothed[i] = smoothed[i-1] - (smoothed[i-1]/period) + values[i]/period
            return smoothed
        
        atr = wilder_smooth(tr, period)
        plus_dm_smooth = wilder_smooth(plus_dm, period)
        minus_dm_smooth = wilder_smooth(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = wilder_smooth(dx, period)
        
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Donchian(20) channels on 6h timeframe
    def donchian_channels(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        middle = (upper + lower) / 2.0
        return upper, lower, middle
    
    donchian_upper, donchian_lower, donchian_middle = donchian_channels(high, low, 20)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Donchian(20) and ADX warmup (ADX needs ~30 for stability)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_adx = adx_12h_aligned[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        curr_middle = donchian_middle[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retraces to Donchian midpoint
            if curr_close <= curr_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retraces to Donchian midpoint
            if curr_close >= curr_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian upper band AND ADX > 25 AND volume confirmation
            if curr_close > curr_upper and curr_adx > 25 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower band AND ADX > 25 AND volume confirmation
            elif curr_close < curr_lower and curr_adx > 25 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals