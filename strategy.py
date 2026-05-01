#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ADX trend filter + volume confirmation.
# Uses Donchian breakouts for clear structure with 1d ADX > 25 to ensure trending markets.
# Volume confirmation (current volume > 1.5x 20-period median) filters low-quality breakouts.
# Works in bull markets (breakouts with trend continuation) and bear markets (breakdowns with trend continuation).
# Discrete sizing 0.25, ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years).
# ADX filter reduces whipsaws in ranging markets, improving Sharpe in both bull and bear regimes.

name = "4h_Donchian20_1dADX25_Volume_v1"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Donchian channels (20-period, using prior bar to avoid look-ahead)
    highest_high_20 = pd.Series(high).shift(1).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).shift(1).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ADX(14) trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate ADX components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1_1d = np.abs(high_1d[1:] - low_1d[1:])
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_first_1d = np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])
    tr_1d = np.concatenate([[tr_first_1d], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    
    # Initial values (simple average of first period)
    atr_1d_init = np.mean(tr_1d[:period])
    plus_dm_init = np.mean(plus_dm[:period])
    minus_dm_init = np.mean(minus_dm[:period])
    
    # Arrays to store smoothed values
    atr_1d_smooth = np.full_like(tr_1d, np.nan)
    plus_dm_smooth = np.full_like(plus_dm, np.nan)
    minus_dm_smooth = np.full_like(minus_dm, np.nan)
    
    # Set initial values
    atr_1d_smooth[period-1] = atr_1d_init
    plus_dm_smooth[period-1] = plus_dm_init
    minus_dm_smooth[period-1] = minus_dm_init
    
    # Wilder's smoothing for remaining values
    for i in range(period, len(tr_1d)):
        atr_1d_smooth[i] = (atr_1d_smooth[i-1] * (period - 1) + tr_1d[i]) / period
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period - 1) + plus_dm[i]) / period
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period - 1) + minus_dm[i]) / period
    
    # Calculate +DI and -DI
    plus_di_1d = 100 * plus_dm_smooth / atr_1d_smooth
    minus_di_1d = 100 * minus_dm_smooth / atr_1d_smooth
    
    # Calculate DX and ADX
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    # ADX is smoothed DX
    adx_1d = np.full_like(dx_1d, np.nan)
    adx_1d[2*period-2] = np.mean(dx_1d[period-1:2*period-1])  # Initial ADX value
    for i in range(2*period-1, len(dx_1d)):
        adx_1d[i] = (adx_1d[i-1] * (period - 1) + dx_1d[i]) / period
    
    # Align ADX to LTF (4h)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Donchian, ADX, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: price > Donchian upper band AND trending AND volume spike
            if curr_close > highest_high_20[i] and trending and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price < Donchian lower band AND trending AND volume spike
            elif curr_close < lowest_low_20[i] and trending and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian lower band OR ADX < 20 (trend weakening)
            elif curr_close < lowest_low_20[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian upper band OR ADX < 20 (trend weakening)
            elif curr_close > highest_high_20[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals