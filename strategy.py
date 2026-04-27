#!/usr/bin/env python3
"""
6h_RSI_EMA_Trend_Filter
Hypothesis: On 6h timeframe, use weekly and daily RSI divergence to identify overbought/oversold conditions, 
combined with 6h EMA crossover for trend confirmation and volume spike for momentum validation.
Long when: RSI weekly < 30 (oversold), RSI daily < 30 (oversold), price crosses above 6h EMA(20), and volume > 1.5x average.
Short when: RSI weekly > 70 (overbought), RSI daily > 70 (overbought), price crosses below 6h EMA(20), and volume > 1.5x average.
Exit when price crosses back over/under EMA or RSI reverts to neutral zone.
Designed to capture mean-reversion bounces in extreme conditions with trend and momentum filters to avoid false signals.
Target: 15-30 trades/year to minimize fee drift while capturing high-probability reversals.
Works in both bull and bear markets by fading extremes in the direction of the higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for RSI
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 14:
        return np.zeros(n)
    
    # Get daily data for RSI
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    # Calculate weekly RSI(14)
    close_weekly = df_weekly['close'].values
    delta_weekly = np.diff(close_weekly, prepend=close_weekly[0])
    gain_weekly = np.where(delta_weekly > 0, delta_weekly, 0)
    loss_weekly = np.where(delta_weekly < 0, -delta_weekly, 0)
    avg_gain_weekly = np.full(len(close_weekly), np.nan)
    avg_loss_weekly = np.full(len(close_weekly), np.nan)
    for i in range(14, len(close_weekly)):
        if i == 14:
            avg_gain_weekly[i] = np.mean(gain_weekly[1:15])
            avg_loss_weekly[i] = np.mean(loss_weekly[1:15])
        else:
            avg_gain_weekly[i] = (avg_gain_weekly[i-1] * 13 + gain_weekly[i]) / 14
            avg_loss_weekly[i] = (avg_loss_weekly[i-1] * 13 + loss_weekly[i]) / 14
    rs_weekly = np.where(avg_loss_weekly != 0, avg_gain_weekly / avg_loss_weekly, 100)
    rsi_weekly = 100 - (100 / (1 + rs_weekly))
    
    # Calculate daily RSI(14)
    close_daily = df_daily['close'].values
    delta_daily = np.diff(close_daily, prepend=close_daily[0])
    gain_daily = np.where(delta_daily > 0, delta_daily, 0)
    loss_daily = np.where(delta_daily < 0, -delta_daily, 0)
    avg_gain_daily = np.full(len(close_daily), np.nan)
    avg_loss_daily = np.full(len(close_daily), np.nan)
    for i in range(14, len(close_daily)):
        if i == 14:
            avg_gain_daily[i] = np.mean(gain_daily[1:15])
            avg_loss_daily[i] = np.mean(loss_daily[1:15])
        else:
            avg_gain_daily[i] = (avg_gain_daily[i-1] * 13 + gain_daily[i]) / 14
            avg_loss_daily[i] = (avg_loss_daily[i-1] * 13 + loss_daily[i]) / 14
    rs_daily = np.where(avg_loss_daily != 0, avg_gain_daily / avg_loss_daily, 100)
    rsi_daily = 100 - (100 / (1 + rs_daily))
    
    # Align weekly and daily RSI to 6h timeframe
    rsi_weekly_aligned = align_htf_to_ltf(prices, df_weekly, rsi_weekly)
    rsi_daily_aligned = align_htf_to_ltf(prices, df_daily, rsi_daily)
    
    # Calculate 6h EMA(20)
    ema_period = 20
    ema = np.full(n, np.nan)
    if n >= ema_period:
        multiplier = 2 / (ema_period + 1)
        ema[ema_period - 1] = np.mean(close[:ema_period])
        for i in range(ema_period, n):
            ema[i] = (close[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    
    # Calculate 6h volume average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(20, 14)  # EMA needs 20, RSI needs 14
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_weekly_aligned[i]) or
            np.isnan(rsi_daily_aligned[i]) or
            np.isnan(ema[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # RSI conditions
        weekly_oversold = rsi_weekly_aligned[i] < 30
        weekly_overbought = rsi_weekly_aligned[i] > 70
        daily_oversold = rsi_daily_aligned[i] < 30
        daily_overbought = rsi_daily_aligned[i] > 70
        
        # EMA crossover
        ema_cross_up = (i > 0 and close[i-1] <= ema[i-1] and price > ema[i])
        ema_cross_down = (i > 0 and close[i-1] >= ema[i-1] and price < ema[i])
        
        # Volume confirmation
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: weekly and daily oversold + EMA cross up + volume
            if weekly_oversold and daily_oversold and ema_cross_up and volume_confirmation:
                signals[i] = 0.25
                position = 1
            # Short: weekly and daily overbought + EMA cross down + volume
            elif weekly_overbought and daily_overbought and ema_cross_down and volume_confirmation:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: EMA cross down or RSI returns to neutral
            if ema_cross_down or (rsi_weekly_aligned[i] > 50 and rsi_daily_aligned[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: EMA cross up or RSI returns to neutral
            if ema_cross_up or (rsi_weekly_aligned[i] < 50 and rsi_daily_aligned[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "6h_RSI_EMA_Trend_Filter"
timeframe = "6h"
leverage = 1.0