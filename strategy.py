#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Bollinger Bands with daily ADX trend filter and volume confirmation
# Long when price closes below weekly BB lower band with ADX > 25 (strong trend) and volume > 1.5x average
# Short when price closes above weekly BB upper band with ADX > 25 and volume > 1.5x average
# Weekly Bollinger Bands provide dynamic support/resistance on higher timeframe
# ADX filter ensures we only trade in strong trending conditions, reducing whipsaws
# Volume confirmation adds conviction to the breakout/breakdown
# Works in bull/bear markets: captures continuation of strong trends filtered by volatility bands
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_weeklyBB_ADX_Volume_Trend_v1"
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
    
    # Calculate weekly Bollinger Bands ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly close for Bollinger Bands
    weekly_close = df_weekly['close'].values
    weekly_ma = pd.Series(weekly_close).rolling(window=20, min_periods=20).mean().values
    weekly_std = pd.Series(weekly_close).rolling(window=20, min_periods=20).std().values
    
    # Bollinger Bands: 2 standard deviations
    bb_upper = weekly_ma + (2 * weekly_std)
    bb_lower = weekly_ma - (2 * weekly_std)
    
    # Align weekly BB levels to 6h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_weekly, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_weekly, bb_lower)
    weekly_ma_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ma)
    
    # Calculate daily ADX for trend filter
    df_daily = get_htf_data(prices, '1d')
    
    if len(df_daily) < 14:
        return np.zeros(n)
    
    # True Range calculation
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_daily - np.roll(high_daily, 1)) > (np.roll(low_daily, 1) - low_daily),
                       np.maximum(high_daily - np.roll(high_daily, 1), 0), 0)
    dm_minus = np.where((np.roll(low_daily, 1) - low_daily) > (high_daily - np.roll(high_daily, 1)),
                        np.maximum(np.roll(low_daily, 1) - low_daily, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align daily ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (00-24 UTC - trade all hours for 6h)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = np.ones(n, dtype=bool)  # Trade all hours
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade when ADX indicates strong trend (> 25)
        strong_trend = adx_aligned[i] > 25
        
        if position == 0 and strong_trend:
            # Long entry: price closes below weekly BB lower band with volume confirmation
            if close[i] < bb_lower_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price closes above weekly BB upper band with volume confirmation
            elif close[i] > bb_upper_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses above weekly middle band (mean reversion signal)
            if close[i] > weekly_ma_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses below weekly middle band (mean reversion signal)
            if close[i] < weekly_ma_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals