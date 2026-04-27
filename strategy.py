# 1d_WeeklyBreakout_Pullback_v2
# Hypothesis: On 1d timeframe, take long positions when price breaks above weekly high with pullback to 20 EMA, and short positions when price breaks below weekly low with pullback to 20 EMA. Uses weekly trend filter (price above/below weekly 50 EMA) and volume confirmation (1.5x average). Designed to capture momentum with pullback entries for better risk-reward, working in both bull and bear markets by following the weekly trend.
# Target: 15-25 trades/year to minimize fee drag while capturing significant moves.

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
    
    # Get weekly data for trend and breakout levels
    df_weekly = get_htf_data(prices, '1w')
    
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly 50 EMA for trend filter
    close_weekly = df_weekly['close'].values
    weekly_ema_period = 50
    weekly_ema = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= weekly_ema_period:
        weekly_ema[weekly_ema_period - 1] = np.mean(close_weekly[:weekly_ema_period])
        multiplier = 2 / (weekly_ema_period + 1)
        for i in range(weekly_ema_period, len(close_weekly)):
            weekly_ema[i] = (close_weekly[i] * multiplier) + (weekly_ema[i-1] * (1 - multiplier))
    
    # Calculate weekly high and low for breakout levels
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Calculate daily 20 EMA for pullback entries
    ema_period = 20
    ema = np.full(n, np.nan)
    if n >= ema_period:
        ema[ema_period - 1] = np.mean(close[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, n):
            ema[i] = (close[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    
    # Calculate daily volume average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align weekly indicators to daily timeframe
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need all indicators
    start_idx = max(20, 50)  # daily EMA needs 20, weekly EMA needs 50
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_high_aligned[i]) or
            np.isnan(weekly_low_aligned[i]) or
            np.isnan(weekly_ema_aligned[i]) or
            np.isnan(ema[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Weekly trend filter
        weekly_uptrend = price > weekly_ema_aligned[i]
        weekly_downtrend = price < weekly_ema_aligned[i]
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        # Pullback to daily 20 EMA (within 1%)
        pullback_to_ema = abs(price - ema[i]) / ema[i] < 0.01
        
        if position == 0:
            # Long: break above weekly high with pullback to EMA, volume, and weekly uptrend
            if weekly_uptrend and volume_confirmation and pullback_to_ema and price > weekly_high_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly low with pullback to EMA, volume, and weekly downtrend
            elif weekly_downtrend and volume_confirmation and pullback_to_ema and price < weekly_low_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below weekly low or weekly trend turns down
            if price < weekly_low_aligned[i] or price <= weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price breaks above weekly high or weekly trend turns up
            if price > weekly_high_aligned[i] or price >= weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "1d_WeeklyBreakout_Pullback_v2"
timeframe = "1d"
leverage = 1.0