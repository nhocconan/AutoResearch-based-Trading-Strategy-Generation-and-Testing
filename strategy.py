#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 Breakout with 4h EMA50 Trend Filter and Volume Spike
- Camarilla pivot levels (R3/S3) provide high-probability breakout zones
- Only trade breakouts in direction of 4h EMA50 trend filter to avoid counter-trend
- Volume confirmation (> 1.8x 24-period MA) ensures breakout validity
- Session filter (08-20 UTC) reduces noise during low-liquidity hours
- Fixed position size 0.20 to control risk and minimize fee churn
- Target: 15-37 trades/year per symbol (60-150 total over 4 years) to avoid fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate typical price for Camarilla pivot (using prior day)
    # Since we don't have daily data aligned, we'll approximate using rolling 24-period (1 day of 1h bars)
    typical_price = (high + low + close) / 3.0
    
    # Calculate Camarilla levels using prior 24-bar high/low/close
    # We need to shift by 24 to use completed prior day's data
    if len(prices) < 25:
        return np.zeros(n)
        
    # Use prior completed 24-bar period for pivot calculation
    prior_high = pd.Series(high).rolling(window=24, min_periods=24).max().shift(24)
    prior_low = pd.Series(low).rolling(window=24, min_periods=24).min().shift(24)
    prior_close = pd.Series(close).rolling(window=24, min_periods=24).last().shift(24)
    
    # Avoid look-ahead by ensuring we only use completed prior day
    pivot = (prior_high + prior_low + prior_close) / 3.0
    range_ = prior_high - prior_low
    
    # Camarilla R3 and S3 levels
    r3 = pivot + (range_ * 1.1 / 4)
    s3 = pivot - (range_ * 1.1 / 4)
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: > 1.8x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(48, 50, 24)  # need prior day data, EMA50_4h, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(prior_high[i]) or np.isnan(prior_low[i]) or np.isnan(prior_close[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND above 4h EMA50 AND volume spike
            if (close[i] > r3[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume[i] > 1.8 * vol_ma[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 AND below 4h EMA50 AND volume spike
            elif (close[i] < s3[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume[i] > 1.8 * vol_ma[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: price returns to pivot level OR volume drops significantly
            exit_signal = False
            if position == 1:
                # Exit long when price < pivot OR volume < 1.2x MA
                if close[i] < pivot[i] or volume[i] < 1.2 * vol_ma[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > pivot OR volume < 1.2x MA
                if close[i] > pivot[i] or volume[i] < 1.2 * vol_ma[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0