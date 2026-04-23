#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation
- Long when price breaks above Camarilla R3 AND close > 1w EMA34 AND volume > 2.0x 20-period average
- Short when price breaks below Camarilla S3 AND close < 1w EMA34 AND volume > 2.0x 20-period average
- Exit when price crosses Camarilla pivot point (mean reversion)
- Uses 1w EMA34 for HTF trend alignment to avoid counter-trend entries
- Volume spike ensures institutional participation and reduces false breakouts
- Designed for both bull and bear markets: trend filter prevents counter-trend entries
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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla pivot point
    pp = (prev_high + prev_low + prev_close) / 3
    # Camarilla R3 and S3 levels
    r3 = pp + (prev_high - prev_low) * 1.1 / 4
    s3 = pp - (prev_high - prev_low) * 1.1 / 4
    
    # AlCamarilla levels to 1d timeframe (already aligned)
    camarilla_pp = pp
    camarilla_r3 = r3
    camarilla_s3 = s3
    
    # Get 1w data for EMA34 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34, 20)  # Need 20 for volume MA, 34 for EMA34, 20 for Camarilla (needs 1d data)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_pp[i]) or 
            np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
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
        
        # Volume confirmation
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
                # Exit long: Price crosses below pivot point
                if close[i] < camarilla_pp[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Price crosses above pivot point
                if close[i] > camarilla_pp[i]:
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