#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 12h EMA34 filter and volume confirmation.
# Elder Ray measures bullish/bearish power relative to EMA, identifying trend strength.
# 12h EMA34 acts as trend filter - only take longs when price > EMA34, shorts when price < EMA34.
# Volume confirmation ensures breakouts have participation.
# Designed for low trade frequency (12-37/year) to minimize fee drag in 6h timeframe.
# Works in bull markets (strong bullish power above EMA) and bear markets (strong bearish power below EMA).
name = "6h_ElderRay_12hEMA34_Volume"
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
    
    # Get 12h data for EMA34 filter (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA34 on 12h close
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 13-period EMA for Elder Ray (using close)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA(13)
    bear_power = low - ema_13   # Bear Power = Low - EMA(13)
    
    # Volume confirmation: 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long: Bull Power > 0 (bullish strength) AND price > 12h EMA34 AND volume confirmation
            long_condition = bull_power[i] > 0 and close[i] > ema_34_12h_aligned[i] and vol_confirm
            if long_condition:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bearish strength) AND price < 12h EMA34 AND volume confirmation
            elif bear_power[i] < 0 and close[i] < ema_34_12h_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power <= 0 OR price crosses below 12h EMA34
            exit_condition = bull_power[i] <= 0 or close[i] < ema_34_12h_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power >= 0 OR price crosses above 12h EMA34
            exit_condition = bear_power[i] >= 0 or close[i] > ema_34_12h_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals