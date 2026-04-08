#!/usr/bin/env python3
# 12h_renko_brick_trend_volume_v1
# Hypothesis: 12h Renko brick direction with volume confirmation and weekly trend filter.
# Uses Renko brick direction (bullish/bearish) for trend identification, volume > 1.3x average for confirmation,
# and weekly EMA trend filter to avoid counter-trend trades. Works in bull markets by catching
# uptrend continuations and in bear markets by catching downtrend continuations.
# Renko bricks filter noise, volume filter reduces false signals, targeting 15-30 trades/year.

name = "12h_renko_brick_trend_volume_v1"
timeframe = "12h"
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
    
    # Calculate ATR for Renko brick size (14-period)
    def calculate_atr(high, low, close, period):
        tr = np.zeros_like(close)
        tr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = np.zeros_like(close)
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (tr[i] + atr[i-1] * (period - 1)) / period
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    brick_size = np.max([np.mean(atr) * 0.5, np.percentile(atr, 20)])  # Adaptive brick size
    
    # Calculate Renko bricks
    renko_direction = np.zeros_like(close)  # 1 for bullish brick, -1 for bearish brick
    brick_price = close[0]  # Starting price
    brick_count = 0
    
    for i in range(1, len(close)):
        price_diff = close[i] - brick_price
        if abs(price_diff) >= brick_size:
            num_bricks = int(price_diff / brick_size)
            if num_bricks > 0:
                renko_direction[i] = 1  # Bullish brick
                brick_price += num_bricks * brick_size
            elif num_bricks < 0:
                renko_direction[i] = -1  # Bearish brick
                brick_price += num_bricks * brick_size  # num_bricks is negative
    
    # Volume filter: 20-period average volume
    vol_ma = np.zeros_like(volume)
    if len(volume) >= 20:
        vol_ma[19:] = np.convolve(volume, np.ones(20)/20, mode='valid')
        vol_ma[:19] = vol_ma[19]  # Fill beginning with first valid value
    else:
        vol_ma[:] = np.mean(volume) if np.mean(volume) > 0 else 1.0
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Weekly EMA (50-period) for higher timeframe trend
    ema_period = 50
    ema_weekly = np.zeros_like(close_weekly)
    if len(close_weekly) >= ema_period:
        ema_weekly[ema_period-1] = np.mean(close_weekly[:ema_period])
        for i in range(ema_period, len(close_weekly)):
            ema_weekly[i] = (close_weekly[i] * 2 + ema_weekly[i-1] * (ema_period - 1)) / (ema_period + 1)
    else:
        ema_weekly[:] = np.mean(close_weekly) if len(close_weekly) > 0 else close[0]
    
    # Align weekly EMA to 12h timeframe
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 20  # Need volume MA and enough price data
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is invalid
        if (np.isnan(renko_direction[i]) or np.isnan(ema_weekly_aligned[i]) or 
            np.isnan(vol_ma[i]) or volume[i] == 0 or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period average
        volume_filter = volume[i] > 1.3 * vol_ma[i]
        
        # Higher timeframe trend filter: price above/below weekly EMA
        uptrend_htf = close[i] > ema_weekly_aligned[i]
        downtrend_htf = close[i] < ema_weekly_aligned[i]
        
        if position == 1:  # Long position
            # Exit if Renko turns bearish or volume fails
            if renko_direction[i] == -1 or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if Renko turns bullish or volume fails
            if renko_direction[i] == 1 or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: bullish Renko brick, volume confirmation, and weekly uptrend
            if (renko_direction[i] == 1 and 
                volume_filter and 
                uptrend_htf):
                position = 1
                signals[i] = 0.25
            # Short entry: bearish Renko brick, volume confirmation, and weekly downtrend
            elif (renko_direction[i] == -1 and 
                  volume_filter and 
                  downtrend_htf):
                position = -1
                signals[i] = -0.25
    
    return signals