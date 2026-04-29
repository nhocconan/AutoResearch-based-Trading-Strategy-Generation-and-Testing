#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Donchian(20) breakout with volume confirmation and 1d trend filter
# Long when: price breaks above Donchian(20) high AND ADX(14) > 25 AND 1d EMA50 uptrend AND volume > 1.5x 20-bar avg
# Short when: price breaks below Donchian(20) low AND ADX(14) > 25 AND 1d EMA50 downtrend AND volume > 1.5x 20-bar avg
# Exit: price crosses opposite Donchian level (20-bar low for longs, high for shorts) OR ADX < 20 (trend weakness)
# Uses discrete position sizing (0.25) to minimize fee churn while capturing trends.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h.
# ADX filters choppy markets, Donchian provides structure, volume confirms institutional participation.
# Works in bull markets (trend continuation) and bear markets (strong downtrends captured via shorts).

name = "6h_ADX_Donchian20_VolumeConfirm_1dTrend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ADX(14) on 6h data
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    # Initialize first smoothed values
    plus_dm_sm = np.zeros(n)
    minus_dm_sm = np.zeros(n)
    tr_sm = np.zeros(n)
    
    plus_dm_sm[0] = plus_dm[0]
    minus_dm_sm[0] = minus_dm[0]
    tr_sm[0] = tr[0]
    
    for i in range(1, n):
        plus_dm_sm[i] = alpha * plus_dm[i] + (1 - alpha) * plus_dm_sm[i-1]
        minus_dm_sm[i] = alpha * minus_dm[i] + (1 - alpha) * minus_dm_sm[i-1]
        tr_sm[i] = alpha * tr[i] + (1 - alpha) * tr_sm[i-1]
    
    # Avoid division by zero
    plus_di = np.where(tr_sm != 0, (plus_dm_sm / tr_sm) * 100, 0)
    minus_di = np.where(tr_sm != 0, (minus_dm_sm / tr_sm) * 100, 0)
    
    # DX and ADX
    di_sum = plus_di + minus_di
    dx = np.where(di_sum != 0, np.abs(plus_di - minus_di) / di_sum * 100, 0)
    
    # ADX: smoothed DX
    adx = np.zeros(n)
    adx[period-1] = dx[period-1]  # First ADX value
    for i in range(period, n):
        adx[i] = alpha * dx[i] + (1 - alpha) * adx[i-1]
    
    # Donchian(20) channels
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_50 = ema_50_1d_aligned[i]
        adx_val = adx[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        
        # Determine 1d trend: price vs EMA50
        uptrend = curr_close > ema_50
        downtrend = curr_close < ema_50
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price crosses below Donchian low OR ADX < 20 (trend weakness)
            if curr_low < lower_channel or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high OR ADX < 20 (trend weakness)
            if curr_high > upper_channel or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND ADX > 25 AND 1d uptrend AND volume confirmation
            if curr_high > upper_channel and adx_val > 25 and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low AND ADX > 25 AND 1d downtrend AND volume confirmation
            elif curr_low < lower_channel and adx_val > 25 and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals