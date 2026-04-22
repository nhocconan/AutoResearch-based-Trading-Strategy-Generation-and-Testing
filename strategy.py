#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with 12-hour trend filter and volume confirmation
    # Elder Ray measures bull/bear power relative to EMA13: Bull = High - EMA13, Bear = Low - EMA13
    # In trending markets: strong bull power + price above EMA13 = long setup
    #                strong bear power + price below EMA13 = short setup
    # 12-hour EMA50 filters trend direction: only take longs in uptrend, shorts in downtrend
    # Volume spike confirms institutional participation in the move
    # Targets ~20-30 trades/year to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 6h data for Elder Ray calculations
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA13 on 6h close for Elder Ray
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power_6h = high_6h - ema13_6h  # Bull power: High - EMA13
    bear_power_6h = low_6h - ema13_6h   # Bear power: Low - EMA13
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 6h timeframe
    bull_power_6h_aligned = align_htf_to_ltf(prices, df_6h, bull_power_6h)
    bear_power_6h_aligned = align_htf_to_ltf(prices, df_6h, bear_power_6h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike filter (20-period on 6h)
    vol_ma20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(bull_power_6h_aligned[i]) or np.isnan(bear_power_6h_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Strong bull power + price above 12h EMA50 (uptrend) + volume spike
            if bull_power_6h_aligned[i] > 0 and close[i] > ema50_12h_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Strong bear power + price below 12h EMA50 (downtrend) + volume spike
            elif bear_power_6h_aligned[i] < 0 and close[i] < ema50_12h_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Power deteriorates or trend reverses vs 12h EMA50
            if position == 1:
                if bull_power_6h_aligned[i] <= 0 or close[i] < ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if bear_power_6h_aligned[i] >= 0 or close[i] > ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_12hEMA50_Volume_Session_v1"
timeframe = "6h"
leverage = 1.0