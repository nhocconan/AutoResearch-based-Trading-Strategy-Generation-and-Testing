#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume_v1
Hypothesis: Trade 1h Camarilla R1/S1 breakouts with 4h EMA50 trend filter and 1d volume spike confirmation.
- Trend filter: 4h EMA50 slope determines regime - bullish if rising, bearish if falling.
- In bullish 4h trend: buy 1h breakouts above R1, sell breakdowns below S1.
- In bearish 4h trend: sell breakdowns below S1, buy bounces above S1.
- Volume confirmation: require 1d volume > 2.0x 20-period average to avoid false breakouts.
- Session filter: only trade 08-20 UTC to reduce noise.
- Position size: 0.20. Target: 60-150 total trades over 4 years = 15-37/year.
- Works in bull: trend-following breakouts. Works in bear: mean reversion at extremes + volume exhaustion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h EMA50 slope for trend direction
    ema_50_slope_4h = np.diff(ema_50_4h, prepend=ema_50_4h[0])
    ema_50_slope_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_slope_4h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume MA for spike confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1h Camarilla pivot levels (using previous 1h OHLC)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    
    # Camarilla levels
    r1 = pivot + (range_ * 1.1 / 12)
    s1 = pivot - (range_ * 1.1 / 12)
    r3 = pivot + (range_ * 1.1 / 4)
    s3 = pivot - (range_ * 1.1 / 4)
    
    # Volume spike confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50(4h) and volume MA(20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_50_slope_4h_aligned[i]) or
            np.isnan(r1[i]) or
            np.isnan(s1[i]) or
            np.isnan(r3[i]) or
            np.isnan(s3[i]) or
            np.isnan(pivot[i]) or
            np.isnan(vol_ma_20[i]) or
            not session_filter[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Determine 4h HTF trend using EMA50 slope
        htf_4h_bullish = ema_50_slope_4h_aligned[i] > 0
        htf_4h_bearish = ema_50_slope_4h_aligned[i] < 0
        
        if position == 0:
            if htf_4h_bullish:
                # Bullish 4h trend: trade breakout continuation
                long_setup = (close[i] > r1[i]) and volume_spike[i]
                short_setup = (close[i] < s1[i]) and volume_spike[i]
            elif htf_4h_bearish:
                # Bearish 4h trend: trade mean reversion at extremes
                long_setup = (close[i] > s1[i]) and (close[i] < s3[i]) and volume_spike[i]  # Oversold bounce
                short_setup = (close[i] < r1[i]) and (close[i] > r3[i]) and volume_spike[i]  # Overbought rejection
            else:
                # Flat 4h trend: no trades
                long_setup = False
                short_setup = False
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit conditions
            if htf_4h_bullish:
                # In bullish 4h trend: exit on trend reversal or touch of S1
                exit_signal = (not htf_4h_bullish) or (close[i] < s1[i])
            else:
                # In bearish/flat 4h trend: exit on mean reversion to pivot or touch of R1
                exit_signal = (close[i] > pivot[i]) or (close[i] > r1[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit conditions
            if htf_4h_bearish:
                # In bearish 4h trend: exit on trend reversal or touch of R1
                exit_signal = htf_4h_bullish or (close[i] > r1[i])
            else:
                # In bullish/flat 4h trend: exit on mean reversion to pivot or touch of S1
                exit_signal = (close[i] < pivot[i]) or (close[i] < s1[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume_v1"
timeframe = "1h"
leverage = 1.0