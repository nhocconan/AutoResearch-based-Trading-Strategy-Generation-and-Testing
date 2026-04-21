#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h CRSI (3-period RSI) with 1d ADX trend filter and volume spike confirmation.
# CRSI combines RSI(3), RSI streak (2-period), and percentile rank for extreme readings.
# Long when CRSI < 15 in uptrend (1d ADX > 25) with volume > 1.8x 20-period average.
# Short when CRSI > 85 in downtrend (1d ADX > 25) with volume confirmation.
# Exit when CRSI crosses back to neutral (40-60) or trend weakens (ADX < 20).
# Targets 20-40 trades/year by requiring extreme CRSI + strong trend + volume.
# Works in bull/bear: CRSI captures mean reversion in trends; ADX filter avoids chop.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX(14) for trend strength filter
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
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate RSI(3) for CRSI
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder smoothing for RSI
    def rsi_wilder(series, period):
        avg_gain = np.full_like(series, np.nan, dtype=float)
        avg_loss = np.full_like(series, np.nan, dtype=float)
        if len(series) >= period:
            avg_gain[period-1] = np.mean(series[:period])
            avg_loss[period-1] = np.mean(series[:period])
            for i in range(period, len(series)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + series[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + series[i]) / period
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_3 = rsi_wilder(gain, 3)
    
    # Calculate RSI streak (2-period consecutive up/down)
    up_streak = np.zeros(n)
    down_streak = np.zeros(n)
    for i in range(1, n):
        if delta[i] > 0:  # up day
            up_streak[i] = up_streak[i-1] + 1
            down_streak[i] = 0
        elif delta[i] < 0:  # down day
            down_streak[i] = down_streak[i-1] + 1
            up_streak[i] = 0
        else:  # unchanged
            up_streak[i] = 0
            down_streak[i] = 0
    
    # RSI of streak (using 2-period RSI on up/down streaks)
    rsi_up_streak = rsi_wilder(up_streak, 2)
    rsi_down_streak = rsi_wilder(down_streak, 2)
    # CRSI uses the streak RSI based on direction
    rsi_streak = np.where(delta >= 0, rsi_up_streak, rsi_down_streak)
    
    # Percentile rank of RSI(3) over 100 periods
    def percentile_rank(series, window):
        rank = np.full_like(series, np.nan, dtype=float)
        for i in range(window-1, len(series)):
            window_data = series[i-window+1:i+1]
            rank[i] = np.sum(window_data <= series[i]) / window * 100
        return rank
    
    rsi_3_percentile = percentile_rank(rsi_3, 100)
    
    # CRSI = average of three components
    crsi = (rsi_3 + rsi_streak + rsi_3_percentile) / 3
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(crsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirm = volume > 1.8 * vol_ma[i]
        
        # Trend filter: strong trend (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            if volume_confirm and strong_trend:
                # Long: CRSI extremely oversold (<15) in uptrend
                if crsi[i] < 15:
                    signals[i] = 0.25
                    position = 1
                # Short: CRSI extremely overbought (>85) in downtrend
                elif crsi[i] > 85:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if CRSI returns to neutral (40-60) or trend weakens
                if crsi[i] > 40 or adx_aligned[i] < 20:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if CRSI returns to neutral (40-60) or trend weakens
                if crsi[i] < 60 or adx_aligned[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_CRSI3_1dADX14_Trend_Volume"
timeframe = "12h"
leverage = 1.0