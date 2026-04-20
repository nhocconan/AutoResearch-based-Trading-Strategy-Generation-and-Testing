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
    
    # Calculate weekly 10-period EMA for trend direction
    close_1w = df_1w['close'].values
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Calculate daily 20-period Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    donch_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # Calculate daily 20-period average volume
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
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
        ema_trend = ema_10_1w_aligned[i]
        donch_high_val = donch_high_1d_aligned[i]
        donch_low_val = donch_low_1d_aligned[i]
        vol_val = prices['volume'].iloc[i]
        vol_avg_val = vol_avg_20_aligned[i]
        atr_val = atr_14_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_trend) or np.isnan(donch_high_val) or 
            np.isnan(donch_low_val) or np.isnan(vol_avg_val) or np.isnan(atr_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Weekly EMA10 up, price breaks above Donchian high, volume above average, volatility sufficient
            if ema_trend > 0 and close_val > donch_high_val and vol_val > vol_avg_val and atr_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: Weekly EMA10 down, price breaks below Donchian low, volume above average, volatility sufficient
            elif ema_trend < 0 and close_val < donch_low_val and vol_val > vol_avg_val and atr_val > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low
            if close_val < donch_low_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high
            if close_val > donch_high_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 1d_WeeklyEMA10_Donchian_Breakout_Volume
# Uses weekly EMA(10) for trend direction filter
# Uses daily Donchian(20) breakouts for entry
# Requires volume confirmation above 20-period average
# Session filter: 8-20 UTC to avoid low-volume periods
# Exits when price breaks opposite Donchian level
# Designed for 1d timeframe with ~10-25 trades/year
name = "1d_WeeklyEMA10_Donchian_Breakout_Volume"
timeframe = "1d"
leverage = 1.0