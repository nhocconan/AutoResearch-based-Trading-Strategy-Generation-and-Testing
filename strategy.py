#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation
# Uses discrete position sizing (0.20) to minimize fee churn. Designed for 1h timeframe to capture
# intraday momentum while using 4h trend for direction and volume for confirmation. Session filter
# (08-20 UTC) reduces noise. Target: 15-30 trades/year per symbol to stay within fee limits.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla pivot levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels: based on previous 1d bar's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla R1 and S1 levels
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    camarilla_r1 = close_1d_prev + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d_prev - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 1h timeframe (using previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get 1h data for volume EMA(20) for volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1h volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20[i])
        
        # 4h trend: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = close[i] > ema_50_4h_aligned[i]
        bearish_trend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + volume confirmation + bullish 4h trend
            if (close[i] > camarilla_r1_aligned[i] and volume_confirmed and bullish_trend):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S1 + volume confirmation + bearish 4h trend
            elif (close[i] < camarilla_s1_aligned[i] and volume_confirmed and bearish_trend):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price falls below Camarilla S1 OR 4h trend turns bearish
            if close[i] < camarilla_s1_aligned[i] or bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price rises above Camarilla R1 OR 4h trend turns bullish
            if close[i] > camarilla_r1_aligned[i] or bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals