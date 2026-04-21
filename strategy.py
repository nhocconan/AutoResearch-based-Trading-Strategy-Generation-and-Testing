#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(30) breakout with 1d ADX(20) trend filter and volume confirmation.
# Long when price breaks above upper Donchian in uptrend (1d ADX > 20), short when breaks below lower Donchian in downtrend.
# Volume > 1.3x 20-period average confirms breakout strength. Exit when price crosses back below/above 10-period SMA.
# Target: 20-30 trades/year by requiring strong trend + volume + breakout alignment.
# Works in bull/bear: ADX filter ensures only strong trends are traded, avoiding whipsaws in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX(20) for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilder_smooth(tr, 20)
    plus_di = 100 * wilder_smooth(plus_dm, 20) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 20) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 20)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 30-period Donchian channels on 12h data
    high_roll = prices['high'].rolling(window=30, min_periods=30).max()
    low_roll = prices['low'].rolling(window=30, min_periods=30).min()
    upper = high_roll.values
    lower = low_roll.values
    
    # Calculate 10-period SMA for exit
    sma_10 = prices['close'].rolling(window=10, min_periods=10).mean().values
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(sma_10[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume > 1.3 * vol_ma[i]
        
        # Trend filter: strong trend (ADX > 20)
        strong_trend = adx_aligned[i] > 20
        
        if position == 0:
            if volume_confirm and strong_trend:
                # Long: price breaks above upper Donchian
                if price > upper[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below lower Donchian
                elif price < lower[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price crosses below 10-period SMA
                if price < sma_10[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price crosses above 10-period SMA
                if price > sma_10[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian30_Breakout_1dADX20_Trend_Volume"
timeframe = "12h"
leverage = 1.0