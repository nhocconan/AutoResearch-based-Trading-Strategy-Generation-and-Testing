#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R4/S4 breakout with 1d EMA50 trend filter and volume spike confirmation
# Uses discrete position sizing (0.25) to minimize fee churn. Designed to work in both bull and bear markets
# by combining mean-reversion pivot breaks with trend filtering. Target: 12-37 trades/year per symbol.

name = "12h_Camarilla_R4S4_Breakout_1dEMA50_VolumeSpike_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1d data for Camarilla pivot levels (based on previous day's OHLC)
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla R4 and S4 levels: based on previous 1d bar's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla R4 and S4 levels
    # R4 = close + 1.1*(high - low)
    # S4 = close - 1.1*(high - low)
    camarilla_r4 = close_1d_prev + 1.1 * (high_1d - low_1d)
    camarilla_s4 = close_1d_prev - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 12h timeframe (using previous day's levels)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Get 12h data for volume EMA(20) for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h volume EMA(20) for volume confirmation
    vol_12h = df_12h['volume'].values
    vol_ema_20 = pd.Series(vol_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 2.0 x 20-period EMA
        volume_confirmed = volume[i] > (2.0 * vol_ema_20_aligned[i])
        
        # 1d trend: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = close[i] > ema_50_1d_aligned[i]
        bearish_trend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R4 + volume confirmation + bullish 1d trend
            if (close[i] > camarilla_r4_aligned[i] and volume_confirmed and bullish_trend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S4 + volume confirmation + bearish 1d trend
            elif (close[i] < camarilla_s4_aligned[i] and volume_confirmed and bearish_trend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below Camarilla S4 OR 1d trend turns bearish
            if close[i] < camarilla_s4_aligned[i] or bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above Camarilla R4 OR 1d trend turns bullish
            if close[i] > camarilla_r4_aligned[i] or bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals