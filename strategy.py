#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above R3 (bullish bias) + price > 4h EMA50 (uptrend) + volume > 1.5x 20-bar average.
# Short when price breaks below S3 (bearish bias) + price < 4h EMA50 (downtrend) + volume > 1.5x 20-bar average.
# Uses ATR trailing stop (2.0x) for risk management.
# Targets 60-150 total trades over 4 years (15-37/year) with discrete position sizing (0.20).
# Camarilla levels provide institutional support/resistance, effective in ranging and trending markets.
# 4h EMA50 filter ensures alignment with higher-timeframe trend, improving performance in bear markets.
# Volume confirmation ensures breakouts have conviction, reducing false signals.
# Session filter (08-20 UTC) reduces noise during low-liquidity periods.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeConfirm_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels using previous day's range
    # Need daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Align previous day's data to 1h timeframe
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Camarilla levels: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    range_hl = prev_high_aligned - prev_low_aligned
    r3 = prev_close_aligned + (range_hl * 1.1 / 4)
    s3 = prev_close_aligned - (range_hl * 1.1 / 4)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # ATR for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(50, 20)  # warmup for EMA50 and Camarilla
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if np.isnan(ema_50_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]):
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            continue
        
        if not in_session[i]:
            # Force flat outside session
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 + uptrend + volume confirmation
            if curr_close > r3[i] and curr_close > ema_50_aligned[i] and curr_volume_confirm:
                signals[i] = 0.20
                position = 1
                highest_since_entry = curr_close
            # Short: price breaks below S3 + downtrend + volume confirmation
            elif curr_close < s3[i] and curr_close < ema_50_aligned[i] and curr_volume_confirm:
                signals[i] = -0.20
                position = -1
                lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_since_entry:
                highest_since_entry = curr_high
            
            # Trailing stop: 2.0 * ATR below highest since entry
            if curr_close < highest_since_entry - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Trailing stop: 2.0 * ATR above lowest since entry
            if curr_close > lowest_since_entry + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals