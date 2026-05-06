#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout with volume confirmation and weekly ADX trend filter
# Long when price breaks above 20-week Donchian upper with volume > 1.5x average and weekly ADX > 25
# Short when price breaks below 20-week Donchian lower with volume > 1.5x average and weekly ADX > 25
# Weekly Donchian provides major trend structure. Volume confirms breakout strength.
# Weekly ADX filter ensures trades align with strong weekly trend, reducing false breakouts.
# Works in bull/bear markets: breakouts capture momentum, trend filter avoids counter-trend trades.
# Target: 20-50 trades per year (80-200 over 4 years) with 0.25 position sizing.

name = "1d_20wDonchian_Breakout_Volume_ADX"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly Donchian channels and ADX ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    high_20w = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian to daily timeframe
    upper_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    lower_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # Weekly ADX for trend strength
    # Calculate +DM, -DM, TR
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    up_move = high_w[1:] - high_w[:-1]
    down_move = low_w[:-1] - low_w[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr1 = high_w[1:] - low_w[1:]
    tr2 = np.abs(high_w[1:] - close_w[:-1])
    tr3 = np.abs(low_w[1:] - close_w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (alpha = 1/period)
    def wilder_smoothing(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] + alpha * (data[i-1] - result[i-1])
        return result
    
    if len(tr) < 20:
        return np.zeros(n)
    
    atr_w = wilder_smoothing(tr, 20)
    plus_di_w = 100 * wilder_smoothing(plus_dm, 20) / atr_w
    minus_di_w = 100 * wilder_smoothing(minus_dm, 20) / atr_w
    dx_w = 100 * np.abs(plus_di_w - minus_di_w) / (plus_di_w + minus_di_w)
    adx_w = wilder_smoothing(dx_w, 20)
    
    # Align weekly ADX to daily timeframe
    adx_w_aligned = align_htf_to_ltf(prices, df_1w, adx_w)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_20w_aligned[i]) or np.isnan(lower_20w_aligned[i]) or 
            np.isnan(adx_w_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above 20-week high with volume and trend confirmation
            if close[i] > upper_20w_aligned[i] and volume_filter[i] and adx_w_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below 20-week low with volume and trend confirmation
            elif close[i] < lower_20w_aligned[i] and volume_filter[i] and adx_w_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below 20-week low or ADX falls below 20
            if close[i] < lower_20w_aligned[i] or adx_w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above 20-week high or ADX falls below 20
            if close[i] > upper_20w_aligned[i] or adx_w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals