#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation
- Long when price breaks above Camarilla R3 AND close > 1w EMA34 AND volume > 2.0x 20-period average
- Short when price breaks below Camarilla S3 AND close < 1w EMA34 AND volume > 2.0x 20-period average
- Exit when price crosses Camarilla pivot point (mean reversion)
- Uses 1w EMA34 for HTF trend alignment to avoid counter-trend entries
- Target: 7-25 trades/year (30-100 total over 4 years) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Camarilla pivot levels from previous day (using 1d data)
    # We need to shift by 1 to avoid look-ahead: use previous day's OHLC
    prev_close = prices['close'].shift(1).values
    prev_high = prices['high'].shift(1).values
    prev_low = prices['low'].shift(1).values
    
    # Camarilla levels calculation
    # Pivot = (prev_high + prev_low + prev_close) / 3
    # Range = prev_high - prev_low
    # R3 = prev_close + Range * 1.1/2
    # S3 = prev_close - Range * 1.1/2
    pivot = (prev_high + prev_low + prev_close) / 3
    rng = prev_high - prev_low
    camarilla_r3 = prev_close + rng * 1.1 / 2
    camarilla_s3 = prev_close - rng * 1.1 / 2
    camarilla_pivot = pivot  # Exit level
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # Need 20 for volume MA, 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_pivot[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla breakout conditions
        breakout_up = close[i] > camarilla_r3[i]  # Break above R3
        breakout_down = close[i] < camarilla_s3[i]  # Break below S3
        
        # Trend filter (using 1w EMA34)
        uptrend = close[i] > ema34_1w_aligned[i]
        downtrend = close[i] < ema34_1w_aligned[i]
        
        # Volume confirmation (stricter threshold)
        volume_ok = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Camarilla breakout up + uptrend + volume confirmation
            if breakout_up and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla breakout down + downtrend + volume confirmation
            elif breakout_down and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses Camarilla pivot point (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below pivot
                if close[i] < camarilla_pivot[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Price crosses above pivot
                if close[i] > camarilla_pivot[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Camarilla_R3S3_1wEMA34_Trend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0