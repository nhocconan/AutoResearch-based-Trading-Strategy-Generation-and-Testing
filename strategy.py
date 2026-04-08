#!/usr/bin/env python3
# 4h_rsi_volume_trend_v1
# Hypothesis: Combine RSI(14) oversold/overbought signals with volume confirmation and daily trend filter.
# RSI < 30 for long, RSI > 70 for short, only when volume > 1.5x 20-period average.
# Trend filter: price above/below daily EMA50 to avoid counter-trend trades.
# Designed for 4h timeframe with ~20-40 trades/year to minimize fee drag.

name = "4h_rsi_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI calculation (14-period)
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    for i in range(rsi_period+1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:rsi_period+1] = 50  # neutral before sufficient data
    
    # Volume moving average (20-period)
    vol_ma = np.zeros_like(volume)
    vol_ma[19:] = np.convolve(volume, np.ones(20)/20, mode='valid')
    vol_ma[:19] = vol_ma[19]
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    close_daily = df_daily['close'].values
    
    # Daily EMA (50-period) for trend filter
    ema_period = 50
    ema_daily = np.zeros_like(close_daily)
    if len(close_daily) >= ema_period:
        ema_daily[ema_period-1] = np.mean(close_daily[:ema_period])
        for i in range(ema_period, len(close_daily)):
            ema_daily[i] = (close_daily[i] * 2 + ema_daily[i-1] * (ema_period - 1)) / (ema_period + 1)
    
    # Align daily EMA to 4h timeframe
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(rsi_period+1, 20, ema_period) + 5
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or volume[i] == 0 or 
            np.isnan(ema_daily_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Higher timeframe trend filter: price above/below daily EMA
        uptrend_htf = close[i] > ema_daily_aligned[i]
        downtrend_htf = close[i] < ema_daily_aligned[i]
        
        if position == 1:  # Long position
            # Exit if RSI exits overbought, trend reverses, or volume fails
            if rsi[i] > 70 or not uptrend_htf or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if RSI exits oversold, trend reverses, or volume fails
            if rsi[i] < 30 or not downtrend_htf or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: RSI oversold, volume confirmation, and daily uptrend
            if rsi_oversold and volume_filter and uptrend_htf:
                position = 1
                signals[i] = 0.25
            # Short entry: RSI overbought, volume confirmation, and daily downtrend
            elif rsi_overbought and volume_filter and downtrend_htf:
                position = -1
                signals[i] = -0.25
    
    return signals