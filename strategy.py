#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d ADX trend filter and volume spike confirmation.
# Long when Williams %R crosses above -80 (oversold bounce) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-bar average.
# Short when Williams %R crosses below -20 (overbought rejection) AND 1d ADX > 25 AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25. Session filter 08-20 UTC.
# Target: 50-150 total trades over 4 years (12-37/year) for BTC/ETH/SOL.
# Williams %R identifies exhaustion points in ranging markets; ADX filters for trending conditions to avoid chop.
# Volume spike confirms conviction behind the reversal. Works in both bull (buy dips) and bear (sell rallies).

name = "6h_WilliamsR_ADX_VolumeReversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ADX calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # prepend NaN for first element
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_14 = wilders_smoothing(tr, 14)
    plus_dm_14 = wilders_smoothing(plus_dm, 14)
    minus_dm_14 = wilders_smoothing(minus_dm, 14)
    
    # DI+ and DI-
    plus_di_14 = np.where(tr_14 != 0, (plus_dm_14 / tr_14) * 100, 0)
    minus_di_14 = np.where(tr_14 != 0, (minus_dm_14 / tr_14) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di_14 + minus_di_14) != 0, 
                  np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14) * 100, 
                  0)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    adx_trending = adx_aligned > 25  # ADX > 25 indicates trending market
    
    # Williams %R calculation (14-period) on 6h data
    def williams_r(high, low, close, period=14):
        highest_high = np.full_like(high, np.nan)
        lowest_low = np.full_like(low, np.nan)
        for i in range(len(high)):
            if i < period - 1:
                highest_high[i] = np.nan
                lowest_low[i] = np.nan
            else:
                highest_high[i] = np.max(high[i-period+1:i+1])
                lowest_low[i] = np.min(low[i-period+1:i+1])
        wr = np.where((highest_high - lowest_low) != 0, 
                      -100 * ((highest_high - close) / (highest_high - lowest_low)), 
                      -50)
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # Williams %R signals: cross above -80 (long) or below -20 (short)
    wr_long_signal = (wr > -80) & (np.concatenate([[False]], wr[:-1]) <= -80)  # crossed above -80
    wr_short_signal = (wr < -20) & (np.concatenate([[False]], wr[:-1]) >= -20)  # crossed below -20
    
    # Volume confirmation: current 6h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for WR, ADX, and volume MA
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if np.isnan(wr[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: WR crosses above -80 AND ADX trending AND volume confirmation
            if (wr_long_signal[i] and 
                adx_trending[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: WR crosses below -20 AND ADX trending AND volume confirmation
            elif (wr_short_signal[i] and 
                  adx_trending[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: WR crosses below -50 (mean reversion) OR ADX loses trend
            if (wr[i] < -50) or (not adx_trending[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: WR crosses above -50 (mean reversion) OR ADX loses trend
            if (wr[i] > -50) or (not adx_trending[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals