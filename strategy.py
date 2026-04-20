#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend direction
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR(14)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilder_smooth(tr, 14)
    
    # Donchian channels (20-period)
    donch_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align all weekly/daily data to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        ema_50_val = ema_50_1w_aligned[i]
        donch_high_val = donch_high_1d_aligned[i]
        donch_low_val = donch_low_1d_aligned[i]
        atr_val = atr_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_50_val) or np.isnan(donch_high_val) or 
            np.isnan(donch_low_val) or np.isnan(atr_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above weekly EMA50 and breaks above daily Donchian high
            if close_val > ema_50_val and close_val > donch_high_val:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA50 and breaks below daily Donchian low
            elif close_val < ema_50_val and close_val < donch_low_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below daily Donchian low or closes below weekly EMA50
            if close_val < donch_low_val or close_val < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above daily Donchian high or closes above weekly EMA50
            if close_val > donch_high_val or close_val > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 1d_WeeklyEMA50_DailyDonchian_Breakout_v1
# Uses weekly EMA(50) for trend filter
# Uses daily Donchian(20) breakouts for entry
# Exits when price breaks opposite Donchian level or trend reverses
# Designed for 1d timeframe with ~10-25 trades/year
name = "1d_WeeklyEMA50_DailyDonchian_Breakout_v1"
timeframe = "1d"
leverage = 1.0