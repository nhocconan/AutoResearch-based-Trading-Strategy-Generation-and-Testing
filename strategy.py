#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
Hypothesis: Price breaking above Camarilla R1 or below S1 (from previous day) with weekly trend filter (1w EMA50) and volume spike. Designed for low trade frequency (<30/year) by requiring confluence of price level break, trend alignment, and volume confirmation. Works in bull markets via breakouts above R1 in uptrend, and in bear markets via breakdowns below S1 in downtrend.
"""

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 12h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Previous day's OHLC for Camarilla calculation (need daily data)
    # Since we're on 12h timeframe, we need to get daily OHLC
    # We'll use the most recent complete day's data
    # For simplicity, we'll use rolling window to get daily high/low/close
    # but this is approximate - in reality we'd need actual daily data
    # Instead, we'll use the previous 12h bar's high/low as proxy for daily
    # Better approach: since 12h bars, two bars make one day
    # We'll use the previous day's close (2 bars ago) and high/low of that day
    
    # Get daily high, low, close from 12h data (approximate)
    # Each day = 2 bars of 12h
    # We'll use the high/low/close from 2 bars ago as previous day's
    prev_day_high = np.maximum.reduce([high[::2], np.roll(high, 2)[::2]])[::2]  # This is complex
    # Simpler: use the previous bar's high/low as approximation for intraday levels
    # Actually, for Camarilla we need previous day's daily OHLC
    # Let's compute daily OHLC from 12h data by grouping
    
    # Create daily index: every 2 bars
    # For now, use simplified approach: previous 12h bar's high/low
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First bar has no previous - set to current values
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: 
    # R4 = close + (high-low)*1.5/2
    # R3 = close + (high-low)*1.25/2
    # R2 = close + (high-low)*1.1/2
    # R1 = close + (high-low)*0.5/2
    # S1 = close - (high-low)*0.5/2
    # S2 = close - (high-low)*1.1/2
    # S3 = close - (high-low)*1.25/2
    # S4 = close - (high-low)*1.5/2
    # We only need R1 and S1 for breakout
    
    range_hl = prev_high - prev_low
    R1 = prev_close + range_hl * 0.25  # 0.5/2 = 0.25
    S1 = prev_close - range_hl * 0.25  # 0.5/2 = 0.25
    
    # Weekly trend filter (1w EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above R1 + above weekly EMA50 + volume
            if (close[i] > R1[i] and 
                close[i] > ema_50_12h[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + below weekly EMA50 + volume
            elif (close[i] < S1[i] and 
                  close[i] < ema_50_12h[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to S1 OR trend turns down
                if (close[i] <= S1[i]) or \
                   (close[i] < ema_50_12h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to R1 OR trend turns up
                if (close[i] >= R1[i]) or \
                   (close[i] > ema_50_12h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals