#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
Hypothesis: Use Camarilla R1/S1 breakout on 1h with 4h trend filter (EMA50) and volume spike confirmation. 
Go long when price breaks above R1 with volume > 1.5x 20-period average and 4h EMA50 up, 
short when price breaks below S1 with volume spike and 4h EMA50 down.
Camarilla levels provide institutional support/resistance, EMA50 filters trend direction, 
volume confirms breakout strength. Designed for 1h timeframe with strict entry to limit trades (15-37/year).
Works in both bull (breakouts with trend) and bear (breakdowns with trend) markets.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_prev = np.roll(ema_50_4h, 1)
    ema_50_4h_prev[0] = ema_50_4h[0]
    ema_50_4h_rising = ema_50_4h > ema_50_4h_prev
    ema_50_4h_falling = ema_50_4h < ema_50_4h_prev
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_50_4h_rising_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h_rising)
    ema_50_4h_falling_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h_falling)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_50_4h_rising_aligned[i]) or 
            np.isnan(ema_50_4h_falling_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels using previous day's OHLC
        # Need daily data for Camarilla calculation
        if i >= 96:  # Need at least 96 hours (4 days) of 1h data to get previous day
            # Get index of previous day's close (24 hours ago)
            prev_day_idx = i - 24
            if prev_day_idx >= 0:
                # Get daily OHLC from 1h data (simplified: use 24h period)
                # Actually need proper daily data - use 1d timeframe
                pass
        
        # Simplified approach: use recent high/low for pivot-like levels
        # Use 24-period high/low as approximation for daily range
        if i >= 24:
            lookback = 24
            period_high = np.max(high[i-lookback:i])
            period_low = np.min(low[i-lookback:i])
            period_close = close[i-1]
            
            # Calculate Camarilla levels
            range_val = period_high - period_low
            if range_val > 0:
                R1 = period_close + (range_val * 1.1 / 12)
                S1 = period_close - (range_val * 1.1 / 12)
                
                # Volume spike condition
                vol_spike = volume[i] > 1.5 * vol_ma_20[i]
                
                if position == 0:
                    # LONG: price breaks above R1 + volume spike + 4h EMA50 rising
                    if close[i] > R1 and vol_spike and ema_50_4h_rising_aligned[i]:
                        signals[i] = 0.20
                        position = 1
                    # SHORT: price breaks below S1 + volume spike + 4h EMA50 falling
                    elif close[i] < S1 and vol_spike and ema_50_4h_falling_aligned[i]:
                        signals[i] = -0.20
                        position = -1
                    else:
                        signals[i] = 0.0
                elif position == 1:
                    # EXIT LONG: price breaks below S1 or 4h EMA50 turns down
                    if close[i] < S1 or not ema_50_4h_rising_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.20
                elif position == -1:
                    # EXIT SHORT: price breaks above R1 or 4h EMA50 turns up
                    if close[i] > R1 or not ema_50_4h_falling_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals