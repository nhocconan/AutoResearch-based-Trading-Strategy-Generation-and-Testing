#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly Donchian(20) breakout with weekly ADX(14) trend filter and volume confirmation.
# Long when price breaks above weekly upper Donchian in weekly uptrend (weekly ADX > 25), short when breaks below lower Donchian in weekly downtrend.
# Volume > 1.5x 20-period average confirms breakout strength. Uses weekly ADX to filter weak trends and avoid chop.
# Target: 10-25 trades/year by requiring strong weekly trend + volume + breakout alignment.
# Works in bull/bear: weekly ADX filter ensures only strong trends are traded, avoiding whipsaws in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly ADX(14) for trend strength filter
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # True Range
    tr1 = np.abs(high_w - low_w)
    tr2 = np.abs(high_w - np.roll(close_w, 1))
    tr3 = np.abs(low_w - np.roll(close_w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    up_move = np.diff(high_w, prepend=high_w[0])
    down_move = -np.diff(low_w, prepend=low_w[0])
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
    
    atr_w = wilder_smooth(tr, 14)
    plus_di_w = 100 * wilder_smooth(plus_dm, 14) / atr_w
    minus_di_w = 100 * wilder_smooth(minus_dm, 14) / atr_w
    dx_w = 100 * np.abs(plus_di_w - minus_di_w) / (plus_di_w + minus_di_w)
    adx_w = wilder_smooth(dx_w, 14)
    
    # Align weekly ADX to daily timeframe
    adx_w_aligned = align_htf_to_ltf(prices, df_weekly, adx_w)
    
    # Calculate weekly Donchian channels (20-period)
    high_roll_w = df_weekly['high'].rolling(window=20, min_periods=20).max()
    low_roll_w = df_weekly['low'].rolling(window=20, min_periods=20).min()
    upper_w = high_roll_w.values
    lower_w = low_roll_w.values
    
    # Align weekly Donchian channels to daily timeframe
    upper_w_aligned = align_htf_to_ltf(prices, df_weekly, upper_w)
    lower_w_aligned = align_htf_to_ltf(prices, df_weekly, lower_w)
    
    # Pre-compute daily volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(adx_w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(upper_w_aligned[i]) or np.isnan(lower_w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Trend filter: strong trend (weekly ADX > 25)
        strong_trend = adx_w_aligned[i] > 25
        
        if position == 0:
            if volume_confirm and strong_trend:
                # Long: price breaks above weekly upper Donchian
                if price > upper_w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below weekly lower Donchian
                elif price < lower_w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price breaks below weekly lower Donchian (failed breakout) or weak trend
                if price < lower_w_aligned[i] or adx_w_aligned[i] < 20:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price breaks above weekly upper Donchian (failed breakdown) or weak trend
                if price > upper_w_aligned[i] or adx_w_aligned[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyDonchian20_WeeklyADX14_Trend_Volume"
timeframe = "1d"
leverage = 1.0