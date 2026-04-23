#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume spike confirmation
- Long when price breaks above 1h Camarilla R3 AND price > 4h EMA34 AND volume > 2.0x 20-period average
- Short when price breaks below 1h Camarilla S3 AND price < 4h EMA34 AND volume > 2.0x 20-period average
- Exit when price crosses the 1h Camarilla midpoint (mean reversion to median)
- Uses 4h EMA34 for HTF trend alignment to avoid counter-trend entries
- Volume spike ensures institutional participation and reduces false breakouts
- Session filter: only trade 08:00-20:00 UTC to avoid low-liquidity periods
- Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag
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
    open_time = prices['open_time']
    
    # Pre-compute session hours for filtering
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data for EMA34 trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate 4h EMA34
    ema34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Get 1h data for Camarilla pivot levels (primary timeframe)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 10:
        return np.zeros(n)
    
    # Calculate 1h Camarilla levels (based on previous bar's OHLC)
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # Using previous 1h bar's OHLC to avoid look-ahead
    prev_high = df_1h['high'].shift(1)
    prev_low = df_1h['low'].shift(1)
    prev_close = df_1h['close'].shift(1)
    
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    camarilla_mid = (camarilla_r3 + camarilla_s3) / 2.0
    
    # Convert to numpy arrays and align to 1h timeframe
    camarilla_r3_vals = camarilla_r3.values
    camarilla_s3_vals = camarilla_s3.values
    camarilla_mid_vals = camarilla_mid.values
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_r3_vals)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_s3_vals)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1h, camarilla_mid_vals)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 35, 21)  # Need 20 for volume MA, 35 for EMA34 (34+1), 1 for Camarilla (shifted)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or 
            np.isnan(ema34_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions (using 1h Camarilla levels)
        breakout_up = close[i] > camarilla_r3_aligned[i]  # Break above Camarilla R3
        breakout_down = close[i] < camarilla_s3_aligned[i]  # Break below Camarilla S3
        
        # Trend filter (using 4h EMA34)
        uptrend = close[i] > ema34_4h_aligned[i]
        downtrend = close[i] < ema34_4h_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: bullish breakout + uptrend + volume confirmation
            if breakout_up and uptrend and volume_ok:
                signals[i] = 0.20
                position = 1
            # Short: bearish breakout + downtrend + volume confirmation
            elif breakout_down and downtrend and volume_ok:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: price crosses 1h Camarilla midpoint (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below midpoint
                if close[i] < camarilla_mid_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses above midpoint
                if close[i] > camarilla_mid_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0