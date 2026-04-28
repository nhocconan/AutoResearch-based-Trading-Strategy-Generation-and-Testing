#!/usr/bin/env python3
"""
4h_RSI_Trend_Breakout_20_1d_RSI14_Volume_Spike
Hypothesis: 4-hour RSI(14) crossing above 60 (bullish momentum) or below 40 (bearish momentum) with 1-day RSI(14) trend filter and volume spike. RSI avoids whipsaw in sideways markets, volume confirms breakout strength, and the daily RSI trend filter ensures alignment with higher timeframe momentum. Target: 25-40 trades/year per symbol.
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
    
    # Get daily data for RSI trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily RSI(14) for trend filter
    close_1d = df_1d['close'].values
    delta_1d = np.diff(close_1d)
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    
    # Wilder's smoothing
    avg_gain_1d = np.full_like(close_1d, np.nan)
    avg_loss_1d = np.full_like(close_1d, np.nan)
    avg_gain_1d[14] = np.nanmean(gain_1d[1:15])
    avg_loss_1d[14] = np.nanmean(loss_1d[1:15])
    
    for i in range(15, len(close_1d)):
        avg_gain_1d[i] = (avg_gain_1d[i-1] * 13 + gain_1d[i]) / 14
        avg_loss_1d[i] = (avg_loss_1d[i-1] * 13 + loss_1d[i]) / 14
    
    rs_1d = np.divide(avg_gain_1d, avg_loss_1d, out=np.full_like(avg_gain_1d, np.nan), where=avg_loss_1d!=0)
    rsi_14_1d = 100 - (100 / (1 + rs_1d))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate RSI(14) on 4h data
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[14] = np.nanmean(gain[1:15])
    avg_loss[14] = np.nanmean(loss[1:15])
    
    for i in range(15, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for all indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_14[i]) or np.isnan(rsi_14_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # RSI conditions
        rsi_bullish = rsi_14[i] > 60  # Bullish momentum
        rsi_bearish = rsi_14[i] < 40  # Bearish momentum
        
        # Trend filter from daily RSI(14)
        daily_bullish = rsi_14_1d_aligned[i] > 50  # Bullish trend on daily
        daily_bearish = rsi_14_1d_aligned[i] < 50  # Bearish trend on daily
        
        # Entry conditions with volume confirmation
        long_entry = rsi_bullish and volume_spike[i] and daily_bullish
        short_entry = rsi_bearish and volume_spike[i] and daily_bearish
        
        # Exit on opposite RSI extreme (reverse position)
        long_exit = rsi_bearish and volume_spike[i]
        short_exit = rsi_bullish and volume_spike[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_RSI_Trend_Breakout_20_1d_RSI14_Volume_Spike"
timeframe = "4h"
leverage = 1.0