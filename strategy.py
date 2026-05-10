#!/usr/bin/env python3
# 4h_10pips_DonchianBreakout_1dTrend_Volume
# Hypothesis: A conservative Donchian breakout strategy with 10-pip buffer to reduce false breakouts, combined with 1d EMA trend filter and volume spike confirmation. Designed to capture real breakouts while avoiding whipsaws in both bull and bear markets. Targets 30-50 trades/year to minimize fee drag.

name = "4h_10pips_DonchianBreakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Donchian channel: 20-period high and low
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_width = high_20 - low_20
    
    # 10-pip buffer (0.001 * price for 4 decimals, scaled to price level)
    buffer = np.where(price > 0, price * 0.0001 * 10, 0)  # 10 pips = 0.0010 in 4-decimal price
    
    # Long breakout: close > (high_20 + buffer)
    long_breakout = close > (high_20 + buffer)
    # Short breakout: close < (low_20 - buffer)
    short_breakout = close < (low_20 - buffer)
    
    # Daily EMA trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 20)  # Warmup for Donchian, daily EMA, volume MA
    
    for i in range(start_idx, n):
        price = close[i]
        
        # Skip if any critical values are NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Recalculate buffer for current price
        buffer = price * 0.0001 * 10  # 10 pips
        
        # Recalculate breakout conditions
        long_breakout = price > (high_20[i] + buffer)
        short_breakout = price < (low_20[i] - buffer)
        
        # Daily trend filter
        uptrend = price > ema_34_1d_aligned[i]
        downtrend = price < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: Donchian breakout + uptrend + volume spike
            if long_breakout and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Donchian breakdown + downtrend + volume spike
            elif short_breakout and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price retrace to midline or trend reversal
            midline = (high_20[i] + low_20[i]) / 2
            if price <= midline or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price retrace to midline or trend reversal
            midline = (high_20[i] + low_20[i]) / 2
            if price >= midline or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals