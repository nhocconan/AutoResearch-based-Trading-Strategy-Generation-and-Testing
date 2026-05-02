#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h trend filter and volume confirmation
# Uses 6h timeframe for signal generation with Camarilla R3/S3 breakouts
# 12h EMA34 provides multi-timeframe trend filter to avoid counter-trend trades
# Volume confirmation (1.8x 20-period average) ensures institutional participation
# Designed for low trade frequency (target: 75-150 total trades over 4 years) to minimize fee drag
# Works in bull markets via trend-aligned breakouts, in bear via strict entry conditions

name = "6h_Camarilla_R3_S3_Breakout_12hEMA34_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Camarilla pivot levels from previous day
    # Need previous day's high, low, close
    prev_high = prices['high'].shift(1).values
    prev_low = prices['low'].shift(1).values
    prev_close = prices['close'].shift(1).values
    
    # Calculate pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Calculate Camarilla levels
    range_hl = prev_high - prev_low
    r3 = pivot + (range_hl * 1.1 / 4.0)
    s3 = pivot - (range_hl * 1.1 / 4.0)
    r4 = pivot + (range_hl * 1.1 / 2.0)
    s4 = pivot - (range_hl * 1.1 / 2.0)
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or
            np.isnan(r4[i]) or np.isnan(s4[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R3 + price > 12h EMA34 + volume confirm
            if close[i] > r3[i] and close[i] > ema_34_12h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + price < 12h EMA34 + volume confirm
            elif close[i] < s3[i] and close[i] < ema_34_12h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below S3 or reverse signal at R4
            if close[i] < s3[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] > r4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above R3 or reverse signal at S4
            if close[i] > r3[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] < s4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals