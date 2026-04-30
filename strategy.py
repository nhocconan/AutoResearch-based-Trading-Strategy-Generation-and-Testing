#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter (EMA34) and volume confirmation.
# Elder Ray = Bull Power (High - EMA13) and Bear Power (Low - EMA13). 
# Long when Bull Power > 0 AND Bear Power rising (less negative) AND price > 1d EMA34 (uptrend) AND volume spike.
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND price < 1d EMA34 (downtrend) AND volume spike.
# Uses discrete position sizing ±0.25 to minimize fee churn. Target: 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets by requiring 1d EMA34 trend alignment to avoid counter-trend whipsaws.

name = "6h_ElderRay_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 6h data ONCE before loop for Elder Ray calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    # Calculate EMA13 on 6h data for Elder Ray
    ema_13_6h = pd.Series(df_6h['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power on 6h
    bull_power = df_6h['high'].values - ema_13_6h  # High - EMA13
    bear_power = df_6h['low'].values - ema_13_6h   # Low - EMA13
    
    # Align Elder Ray components to primary timeframe (6h -> 6h: identity but using helper for consistency)
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 34, 20) + 1  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_bull = bull_power_aligned[i]
        curr_bear = bear_power_aligned[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        curr_close = close[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 (strong buying) AND Bear Power rising (less negative) AND uptrend AND volume
            if (curr_bull > 0 and 
                i > start_idx and bear_power_aligned[i] > bear_power_aligned[i-1] and  # Bear Power rising
                curr_close > curr_ema_34_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (strong selling) AND Bull Power falling (less positive) AND downtrend AND volume
            elif (curr_bear < 0 and 
                  i > start_idx and bull_power_aligned[i] < bull_power_aligned[i-1] and  # Bull Power falling
                  curr_close < curr_ema_34_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position - exit when Bull Power turns negative
            if curr_bull <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit when Bear Power turns positive
            if curr_bear >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals