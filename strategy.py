# 6h Weekly Pivot Breakout with Volume Confirmation and ADX Trend Filter
# Hypothesis: In trending markets (ADX >= 25), weekly pivot levels act as breakout levels - 
# price breaking above R1 or below S1 with volume continues the trend. In ranging markets (ADX < 25),
# price reverts to the weekly pivot point. Weekly pivots derived from prior week's OHLC provide
# institutional reference points. Volume confirms institutional participation. Designed for 6h timeframe
# to capture fewer, higher-quality trades (target: 20-40 trades/year) minimizing fee drag.
# Works in bull markets (breakouts above R1) and bear markets (breakdowns below S1).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot points from prior week's OHLC."""
    if high <= 0 or low <= 0 or close <= 0:
        return close, close, close, close, close, close
    pp = (high + low + close) / 3.0
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    return pp, r1, s1, r2, s2

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Get daily data for ADX trend filter
    df_daily = get_htf_data(prices, '1d')
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate weekly pivot points (using prior week's data)
    pp = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    
    for i in range(len(df_weekly)):
        pp_val, r1_val, s1_val, r2_val, s2_val = calculate_weekly_pivot(
            high_weekly[i], low_weekly[i], close_weekly[i]
        )
        # These values are valid for the entire following week
        # We'll align them to 6h timeframe with proper delay
        pass
    
    # Calculate ADX on daily data for trend filter
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        if n < period * 2:
            return np.full(n, np.nan)
        
        # True Range
        tr = np.maximum(high[1:] - low[1:], 
                       np.maximum(np.abs(high[1:] - close[:-1]), 
                                  np.abs(low[1:] - close[:-1])))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smoothed values (Wilder's smoothing)
        atr = np.full(n, np.nan)
        plus_dm_smooth = np.full(n, np.nan)
        minus_dm_smooth = np.full(n, np.nan)
        
        if n >= period:
            atr[period-1] = np.nanmean(tr[1:period+1])
            plus_dm_smooth[period-1] = np.nanmean(plus_dm[1:period+1])
            minus_dm_smooth[period-1] = np.nanmean(minus_dm[1:period+1])
            
            for i in range(period, n):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
                minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = np.full(n, np.nan)
        minus_di = np.full(n, np.nan)
        dx = np.full(n, np.nan)
        
        for i in range(period, n):
            if atr[i] != 0:
                plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
                minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # ADX
        adx = np.full(n, np.nan)
        if n >= 2*period-1:
            adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
            for i in range(2*period-1, n):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_daily = calculate_adx(high_daily, low_daily, close_daily, 14)
    
    # Calculate volume moving average (24-period for 6h = 6 days)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    
    # Align weekly pivots and daily ADX to 6h timeframe
    # For weekly pivots, we need to use the prior week's values
    # Create arrays of weekly pivot values shifted by one week to avoid look-ahead
    pp_weekly = np.full(len(df_weekly), np.nan)
    r1_weekly = np.full(len(df_weekly), np.nan)
    s1_weekly = np.full(len(df_weekly), np.nan)
    r2_weekly = np.full(len(df_weekly), np.nan)
    s2_weekly = np.full(len(df_weekly), np.nan)
    
    for i in range(1, len(df_weekly)):  # Start from 1 to use prior week
        pp_val, r1_val, s1_val, r2_val, s2_val = calculate_weekly_pivot(
            high_weekly[i-1], low_weekly[i-1], close_weekly[i-1]
        )
        pp_weekly[i] = pp_val
        r1_weekly[i] = r1_val
        s1_weekly[i] = s1_val
        r2_weekly[i] = r2_val
        s2_weekly[i] = s2_val
    
    # Align to 6h timeframe
    pp_6h = align_htf_to_ltf(prices, df_weekly, pp_weekly)
    r1_6h = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_6h = align_htf_to_ltf(prices, df_weekly, s1_weekly)
    r2_6h = align_htf_to_ltf(prices, df_weekly, r2_weekly)
    s2_6h = align_htf_to_ltf(prices, df_weekly, s2_weekly)
    adx_6h = align_htf_to_ltf(prices, df_daily, adx_daily)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need indicators ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(adx_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 24-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX threshold
        trending = adx_6h[i] >= 25
        ranging = adx_6h[i] < 25
        
        if position == 0:
            if ranging:
                # In ranging markets, revert to weekly pivot (PP)
                if close[i] <= pp_6h[i] * 1.002 and vol_confirmed:  # Near PP with volume
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= pp_6h[i] * 0.998 and vol_confirmed:  # Near PP with volume
                    signals[i] = -0.25
                    position = -1
            else:
                # In trending markets, trade breakouts of R1/S1
                if close[i] > r1_6h[i] and vol_confirmed:  # Break above R1
                    signals[i] = 0.25
                    position = 1
                elif close[i] < s1_6h[i] and vol_confirmed:  # Break below S1
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price reaches R2 or returns to PP
            if close[i] >= r2_6h[i] or close[i] <= pp_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches S2 or returns to PP
            if close[i] <= s2_6h[i] or close[i] >= pp_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Breakout_ADX_Volume"
timeframe = "6h"
leverage = 1.0