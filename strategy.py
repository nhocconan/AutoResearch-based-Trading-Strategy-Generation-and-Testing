#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(2) mean reversion with 4h trend filter and volume confirmation
# RSI(2) < 10 for long, > 90 for short captures extreme short-term oversold/overbought conditions.
# 4h EMA50 trend filter ensures trades align with intermediate-term trend to avoid counter-trend whipsaws.
# Volume confirmation (1.5x 20-period EMA) filters low-quality signals.
# Designed for 60-150 total trades over 4 years (15-37/year) with discrete sizing to minimize fee drag.
# Session filter (08-20 UTC) reduces noise during low-liquidity hours.

name = "1h_RSI2_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate RSI(2) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]) or np.isnan(data[i]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    gain_smoothed = wilders_smoothing(gain, 2)
    loss_smoothed = wilders_smoothing(loss, 2)
    
    rs = gain_smoothed / loss_smoothed
    rs = np.where(loss_smoothed == 0, 0, rs)
    rsi_2 = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 20-period EMA on 1h
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid RSI and volume EMA
        # Skip if any value is NaN or outside session
        if (np.isnan(rsi_2[i]) or np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Trend filter: price above/below 4h EMA50
        price_above_ema = close[i] > ema_50_4h_aligned[i]
        price_below_ema = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: RSI(2) < 10 (extreme oversold) + price above 4h EMA50 + volume spike
            if rsi_2[i] < 10 and price_above_ema and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: RSI(2) > 90 (extreme overbought) + price below 4h EMA50 + volume spike
            elif rsi_2[i] > 90 and price_below_ema and volume_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI(2) > 50 (mean reversion complete) or loses trend alignment
            if rsi_2[i] > 50 or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI(2) < 50 (mean reversion complete) or loses trend alignment
            if rsi_2[i] < 50 or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals