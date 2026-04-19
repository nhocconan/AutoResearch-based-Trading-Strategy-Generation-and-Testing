#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + Weekly Trend Filter
# Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Use weekly EMA(34) for trend: price > EMA = bullish trend, price < EMA = bearish trend
# In bullish weekly trend: enter long when Bull Power > 0 and rising
# In bearish weekly trend: enter short when Bear Power > 0 and rising
# Volume confirmation: volume > 1.5x 20-period average
# Target: 20-40 trades/year per symbol to stay within frequency limits
name = "6h_ElderRay_WeeklyTrend_Volume"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(34) for trend
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily EMA(13) for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Elder Ray components
    bull_power = high - ema_13_1d_aligned  # High - EMA(13)
    bear_power = ema_13_1d_aligned - low   # EMA(13) - Low
    
    # Smooth power values (2-period) to reduce noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=2, adjust=False).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=2, adjust=False).mean().values
    
    # Get 6h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 13, 20)  # Ensure weekly EMA, daily EMA, and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(bull_power_smooth[i]) or 
            np.isnan(bear_power_smooth[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        bull_val = bull_power_smooth[i]
        bear_val = bear_power_smooth[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Weekly trend determination
        weekly_bullish = price > ema_34_1w_val
        weekly_bearish = price < ema_34_1w_val
        
        if position == 0:
            # Determine entry based on weekly trend
            if weekly_bullish and volume_confirmed:
                # Bullish weekly trend: look for long signals
                if bull_val > 0 and bull_val > bull_power_smooth[i-1]:
                    signals[i] = 0.25
                    position = 1
            elif weekly_bearish and volume_confirmed:
                # Bearish weekly trend: look for short signals
                if bear_val > 0 and bear_val > bear_power_smooth[i-1]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: weekly trend turns bearish or Bull Power turns negative
            if not weekly_bullish or bull_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly trend turns bullish or Bear Power turns negative
            if not weekly_bearish or bear_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals