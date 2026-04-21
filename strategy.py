#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R3/S3 breakout with volume confirmation and 1d trend filter.
# Go long when price breaks above Camarilla R3 and volume > 1.5x 20-period average.
# Go short when price breaks below Camarilla S3 and volume > 1.5x 20-period average.
# Only take trades in direction of 1d EMA50 trend (long when price > EMA50, short when price < EMA50).
# Uses 1d EMA50 for trend filter to avoid counter-trend trades.
# Target: 20-150 total trades over 4 years by requiring trend alignment + breakout + volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_d = df_1d['close'].values
    ema50_d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_d)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels from previous day
        # Need previous day's high, low, close
        if i < 96:  # Need at least 96 4h bars (4 days) to get previous day
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Get previous day's OHLC (assuming 24*4 = 96 bars per day)
        prev_day_start = i - 96
        prev_day_end = i - 24  # Previous day's close is 24 bars ago
        if prev_day_start < 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        prev_high = np.max(prices['high'].iloc[prev_day_start:prev_day_end+1].values)
        prev_low = np.min(prices['low'].iloc[prev_day_start:prev_day_end+1].values)
        prev_close = prices['close'].iloc[prev_day_end]
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        camarilla_r3 = prev_close + (range_val * 1.1 / 4)
        camarilla_s3 = prev_close - (range_val * 1.1 / 4)
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Calculate 20-period volume average
        vol_lookback_start = max(0, i - 19)
        vol_window = prices['volume'].iloc[vol_lookback_start:i+1].values
        vol_ma_20 = np.mean(vol_window)
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma_20
        
        # Trend filter: price vs daily EMA50
        bull_trend = price > ema50_1d_aligned[i]
        bear_trend = price < ema50_1d_aligned[i]
        
        if position == 0:
            # Enter long on breakout above Camarilla R3 with volume and bullish trend
            if price > camarilla_r3 and volume_confirm and bull_trend:
                signals[i] = 0.25
                position = 1
            # Enter short on breakout below Camarilla S3 with volume and bearish trend
            elif price < camarilla_s3 and volume_confirm and bear_trend:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through Camarilla opposite level
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Camarilla S3
                if price < camarilla_s3:
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above Camarilla R3
                if price > camarilla_r3:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0