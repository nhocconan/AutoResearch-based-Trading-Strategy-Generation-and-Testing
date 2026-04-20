#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d < 50):
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Wilder's smoothing for ATR
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilder_smooth(tr, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily Donchian channels (20-period)
    donch_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # Calculate 40-period EMA for trend filter on daily
    ema_40_1d = pd.Series(close_1d).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_40_1d)
    
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
        atr_val = atr_1d_aligned[i]
        donch_high_val = donch_high_1d_aligned[i]
        donch_low_val = donch_low_1d_aligned[i]
        ema_40_val = ema_40_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(atr_val) or np.isnan(donch_high_val) or 
            np.isnan(donch_low_val) or np.isnan(ema_40_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above EMA40 (uptrend) and breaks above Donchian high with volatility filter
            if close_val > ema_40_val and close_val > donch_high_val and atr_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: price below EMA40 (downtrend) and breaks below Donchian low with volatility filter
            elif close_val < ema_40_val and close_val < donch_low_val and atr_val > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or crosses below EMA40
            if close_val < donch_low_val or close_val < ema_40_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or crosses above EMA40
            if close_val > donch_high_val or close_val > ema_40_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 4h_DailyEMA40_Donchian_Breakout_VolatilityFilter_Session_v1
# Uses daily EMA(40) for trend filter
# Uses daily Donchian(20) breakouts for entry
# Requires positive volatility (ATR > 0) to avoid dead markets
# Session filter: 8-20 UTC to avoid low-volume periods
# Exits when price breaks opposite Donchian level or trend changes (crosses EMA40)
# Designed for 4h timeframe with ~20-40 trades/year
name = "4h_DailyEMA40_Donchian_Breakout_VolatilityFilter_Session_v1"
timeframe = "4h"
leverage = 1.0