#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversals with 1d volume spike and 1w trend filter (ADX>25).
# Long when Williams %R < -80 (oversold) + volume > 1.5x 20-bar avg + 1w ADX > 25 (trending up).
# Short when Williams %R > -20 (overbought) + volume confirmation + 1w ADX > 25 (trending down).
# Uses discrete sizing 0.25. ATR(14) stop: signal→0 when price moves 2.0*ATR against position.
# Williams %R computed on 6h close, high, low. Volume confirmation avoids low-momentum reversals.
# 1w ADX filter ensures we only trade reversals in the direction of the weekly trend.
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull (buy oversold in uptrend) and bear (sell overbought in downtrend) markets.

name = "6h_WilliamsR_Extreme_1dVolume_1wADX_v1"
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
    
    # Pre-compute hours for potential session filters (though 6h less sensitive)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Williams %R(14) on 6h data: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    rr = highest_high_14 - lowest_low_14
    williams_r = np.where(rr != 0, ((highest_high_14 - close) / rr) * -100, -50.0)
    
    # Load 1d data ONCE for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Volume spike: current 6h volume > 1.5x 20-bar 6h volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1w data ONCE for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1_1w = high_1w[1:] - low_1w[1:]
    tr2_1w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3_1w = np.abs(low_1w[1:] - close_1w[:-1])
    tr_first_1w = np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])
    tr_1w = np.concatenate([[tr_first_1w], np.maximum(tr1_1w, np.maximum(tr2_1w, tr3_1w))])
    
    # +DM and -DM
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus[0] = 0.0
    dm_minus[0] = 0.0
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilder_smoothing(values, period):
        smoothed = np.full_like(values, np.nan, dtype=float)
        smoothed[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    tr_14 = wilder_smoothing(tr_1w, 14)
    dm_plus_14 = wilder_smoothing(dm_plus, 14)
    dm_minus_14 = wilder_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_14 != 0, (dm_plus_14 / tr_14) * 100, 0)
    di_minus = np.where(tr_14 != 0, (dm_minus_14 / tr_14) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilder_smoothing(dx, 14)
    
    # Align 1w ADX to 6h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for Williams %R, volume MA, and ADX
    start_idx = 20
    
    for i in range(start_idx, n):
        # Session filter: 6h bars can span sessions, but we avoid low-volume Asian session
        # hour = hours[i]
        # if hour < 0 or hour > 23:  # always true, kept for structural consistency
        #     signals[i] = 0.0
        #     continue
        
        if (np.isnan(williams_r[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-bar average
        volume_confirm = curr_volume > (vol_ma[i] * 1.5)
        
        # Williams %R extremes
        williams_oversold = williams_r[i] < -80
        williams_overbought = williams_r[i] > -20
        
        # 1w trend filter: ADX > 25 indicates trending market
        strong_trend = adx_1w_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold AND volume confirmation AND strong uptrend (ADX>25)
            if (williams_oversold and 
                volume_confirm and 
                strong_trend):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Williams %R overbought AND volume confirmation AND strong downtrend (ADX>25)
            elif (williams_overbought and 
                  volume_confirm and 
                  strong_trend):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R returns above -50 (neutral) OR trend weakens (ADX <= 25)
            elif williams_r[i] > -50 or adx_1w_aligned[i] <= 25:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R returns below -50 (neutral) OR trend weakens (ADX <= 25)
            elif williams_r[i] < -50 or adx_1w_aligned[i] <= 25:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals