#!/usr/bin/env python3
# 4h_Camarilla_R3S3_Breakout_1dEMA34_ADX25_Volume
# Hypothesis: 4h Camarilla R3/S3 breakout in trending markets (ADX>25), filtered by 1d EMA34 trend and volume spike.
# Works in bull/bear by requiring trend alignment.
# Camarilla provides precise levels, ADX filters ranging markets, EMA34 ensures higher timeframe trend alignment.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_ADX25_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    # Wilder's smoothing
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1])), 
                    np.abs(low[1:] - close[:-1]))
    
    # Initialize arrays
    plus_di = np.full(len(high), np.nan)
    minus_di = np.full(len(high), np.nan)
    dx = np.full(len(high), np.nan)
    adx = np.full(len(high), np.nan)
    
    # First TR average (period)
    tr_sum = np.nansum(tr[:period])
    if tr_sum == 0:
        return adx
    
    plus_dm_sum = np.nansum(plus_dm[:period])
    minus_dm_sum = np.nansum(minus_dm[:period])
    
    plus_di[period] = 100 * plus_dm_sum / tr_sum
    minus_di[period] = 100 * minus_dm_sum / tr_sum
    dx[period] = 100 * np.abs(plus_di[period] - minus_di[period]) / (plus_di[period] + minus_di[period])
    
    # Wilder smoothing
    for i in range(period+1, len(high)):
        tr_sum = tr_sum - tr_sum/period + tr[i-1]
        plus_dm_sum = plus_dm_sum - plus_dm_sum/period + plus_dm[i-1]
        minus_dm_sum = minus_dm_sum - minus_dm_sum/period + minus_dm[i-1]
        
        plus_di[i] = 100 * plus_dm_sum / tr_sum
        minus_di[i] = 100 * minus_dm_sum / tr_sum
        dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        if i < 2*period:
            adx[i] = np.nansum(dx[period:i+1]) / (i - period + 1)
        else:
            adx[i] = adx[i-1] - (adx[i-1] - dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # need enough for ADX and other calcs
        return np.zeros(n)
    
    # Get 1d data for Camarilla and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Calculate Camarilla levels from daily data ---
    # For each 4h bar, use previous day's H,L,C
    # We'll shift the daily data by 1 to get previous day
    # But need to align properly
    
    # Get daily OHLC
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R3 = Close + 1.25*(High-Low)
    # S3 = Close - 1.25*(High-Low)
    # Pivot = (High+Low+Close)/3
    camarilla_r3 = daily_close + 1.25 * (daily_high - daily_low)
    camarilla_s3 = daily_close - 1.25 * (daily_high - daily_low)
    camarilla_pivot = (daily_high + daily_low + daily_close) / 3
    
    # Align to 4h: each daily value applies to all 4h bars of that day
    # But we want to use previous day's levels for current day's trading
    # So shift the daily arrays by 1 (to get previous day) then align
    camarilla_r3_prev = np.roll(camarilla_r3, 1)
    camarilla_s3_prev = np.roll(camarilla_s3, 1)
    camarilla_pivot_prev = np.roll(camarilla_pivot, 1)
    # First day will have NaN after roll, which is correct
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_prev)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_prev)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot_prev)
    
    # --- 1d EMA34 trend ---
    close_1d = df_1d['close'].values
    # Calculate EMA with pandas for simplicity and correct handling
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_slope_1d = np.diff(ema_1d, prepend=np.nan)  # today - yesterday
    
    # Align EMA and slope to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_slope_1d)
    
    # --- ADX(14) on 4h ---
    adx_values = calculate_adx(high, low, close, 14)
    
    # --- Volume confirmation ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough for Camarilla (need previous day), EMA34, ADX, vol MA
    start_idx = max(34, 34, 14*2, 20)  # rough estimate
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(ema_slope_1d_aligned[i]) or
            np.isnan(adx_values[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Conditions
        price_above_r3 = close[i] > camarilla_r3_aligned[i]
        price_below_s3 = close[i] < camarilla_s3_aligned[i]
        adx_trending = adx_values[i] > 25
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            if price_above_r3 and ema_slope_1d_aligned[i] > 0 and adx_trending and vol_spike:
                # Long: break above R3, uptrend, volume spike
                signals[i] = 0.25
                position = 1
            elif price_below_s3 and ema_slope_1d_aligned[i] < 0 and adx_trending and vol_spike:
                # Short: break below S3, downtrend, volume spike
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price falls to pivot OR trend weakens (ADX < 20 or EMA slope negative)
                if close[i] < camarilla_pivot_aligned[i] or ema_slope_1d_aligned[i] < 0 or adx_values[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises to pivot OR trend weakens
                if close[i] > camarilla_pivot_aligned[i] or ema_slope_1d_aligned[i] > 0 or adx_values[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals