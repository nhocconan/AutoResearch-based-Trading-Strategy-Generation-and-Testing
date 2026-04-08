#!/usr/bin/env python3
# 1h_momentum_reversal_4h1d_trend_v1
# Hypothesis: 1h momentum reversal with 4h/1d trend filter.
# Long when 1h RSI crosses above 30 and price above 4h EMA20 and 1d EMA50.
# Short when 1h RSI crosses below 70 and price below 4h EMA20 and 1d EMA50.
# Uses session filter (08-20 UTC) and volume confirmation to reduce false signals.
# Targets 15-37 trades/year by requiring multi-timeframe alignment.

name = "1h_momentum_reversal_4h1d_trend_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h RSI (14-period)
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # 1h volume moving average (20-period)
    vol_ma = np.zeros_like(volume)
    vol_ma[19:] = np.convolve(volume, np.ones(20)/20, mode='valid')
    vol_ma[:19] = vol_ma[19]  # Fill beginning with first valid value
    
    # Get 4h data for EMA20
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA (20-period)
    ema_period_4h = 20
    ema_4h = np.zeros_like(close_4h)
    ema_4h[ema_period_4h-1] = np.mean(close_4h[:ema_period_4h])
    for i in range(ema_period_4h, len(close_4h)):
        ema_4h[i] = (close_4h[i] * 2 + ema_4h[i-1] * (ema_period_4h - 1)) / (ema_period_4h + 1)
    
    # Align 4h EMA to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for EMA50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA (50-period)
    ema_period_1d = 50
    ema_1d = np.zeros_like(close_1d)
    ema_1d[ema_period_1d-1] = np.mean(close_1d[:ema_period_1d])
    for i in range(ema_period_1d, len(close_1d)):
        ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * (ema_period_1d - 1)) / (ema_period_1d + 1)
    
    # Align 1d EMA to 1h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(14, 19, 20, 50) + 5
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(rsi[i-1]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period average
        volume_filter = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filters: price above/both EMAs
        above_both_emas = close[i] > ema_4h_aligned[i] and close[i] > ema_1d_aligned[i]
        below_both_emas = close[i] < ema_4h_aligned[i] and close[i] < ema_1d_aligned[i]
        
        # RSI crossover signals
        rsi_cross_above_30 = rsi[i-1] <= 30 and rsi[i] > 30
        rsi_cross_below_70 = rsi[i-1] >= 70 and rsi[i] < 70
        
        if position == 1:  # Long position
            # Exit if RSI crosses below 70 or trend fails or volume fails
            if rsi_cross_below_70 or not above_both_emas or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit if RSI crosses above 30 or trend fails or volume fails
            if rsi_cross_above_30 or not below_both_emas or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: RSI crosses above 30, price above both EMAs, volume confirmation
            if rsi_cross_above_30 and above_both_emas and volume_filter:
                position = 1
                signals[i] = 0.20
            # Short entry: RSI crosses below 70, price below both EMAs, volume confirmation
            elif rsi_cross_below_70 and below_both_emas and volume_filter:
                position = -1
                signals[i] = -0.20
    
    return signals