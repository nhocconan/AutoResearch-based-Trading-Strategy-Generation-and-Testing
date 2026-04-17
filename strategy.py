#!/usr/bin/env python3
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
    
    # === 1w High/Low for range identification (weekly range) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly range position: where current price sits in weekly range
    weekly_range = high_1w - low_1w
    # Avoid division by zero
    weekly_range_safe = np.where(weekly_range == 0, 1, weekly_range)
    weekly_position = (close_1w - low_1w) / weekly_range_safe  # 0 at low, 1 at high
    
    # === 1d Close for daily trend and momentum ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 20-period EMA for daily trend
    ema_20 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 20:
        ema_20[19] = np.mean(close_1d[:20])
        alpha = 2 / (20 + 1)
        for i in range(20, len(close_1d)):
            ema_20[i] = alpha * close_1d[i] + (1 - alpha) * ema_20[i-1]
    else:
        for i in range(len(close_1d)):
            ema_20[i] = np.mean(close_1d[:i+1]) if i >= 0 else close_1d[0]
    
    # Calculate daily price change momentum (1-day return)
    daily_return = np.full_like(close_1d, np.nan)
    for i in range(1, len(close_1d)):
        if close_1d[i-1] != 0:
            daily_return[i] = (close_1d[i] - close_1d[i-1]) / close_1d[i-1]
        else:
            daily_return[i] = 0
    
    # === Align indicators to daily timeframe ===
    weekly_position_aligned = align_htf_to_ltf(prices, df_1w, weekly_position)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    daily_return_aligned = align_htf_to_ltf(prices, df_1d, daily_return)
    
    # === Daily Volume confirmation ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    vol_confirm = volume > vol_ma_20 * 1.5
    
    # === Signal parameters ===
    # Weekly range boundaries for mean reversion
    WEAK_LONG_THRESHOLD = 0.2   # Near weekly low (oversold)
    WEAK_SHORT_THRESHOLD = 0.8  # Near weekly high (overbought)
    MOMENTUM_THRESHOLD = 0.02   # 2% daily momentum
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_position_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or 
            np.isnan(daily_return_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Near weekly low AND positive daily momentum AND above daily EMA20
            if (weekly_position_aligned[i] < WEAK_LONG_THRESHOLD and
                daily_return_aligned[i] > MOMENTUM_THRESHOLD and
                close[i] > ema_20_aligned[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Near weekly high AND negative daily momentum AND below daily EMA20
            elif (weekly_position_aligned[i] > WEAK_SHORT_THRESHOLD and
                  daily_return_aligned[i] < -MOMENTUM_THRESHOLD and
                  close[i] < ema_20_aligned[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Weekly position > 0.5 (middle) OR momentum turns negative OR below EMA20
            if (weekly_position_aligned[i] > 0.5 or
                daily_return_aligned[i] < -MOMENTUM_THRESHOLD/2 or
                close[i] < ema_20_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Weekly position < 0.5 (middle) OR momentum turns positive OR above EMA20
            if (weekly_position_aligned[i] < 0.5 or
                daily_return_aligned[i] > MOMENTUM_THRESHOLD/2 or
                close[i] > ema_20_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyRange_Momentum_EMA20_VolumeFilter"
timeframe = "1d"
leverage = 1.0