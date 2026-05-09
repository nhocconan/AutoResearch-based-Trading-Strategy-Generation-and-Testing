#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_R3S3_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter and pivot levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 4h close for trend filter
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Calculate Camarilla levels from previous 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_vals = df_4h['close'].values
    
    # Camarilla R3, S3 levels: (H-L)*1.1/6
    camarilla_range = (high_4h - low_4h) * 1.1 / 6
    r3_level = close_4h_vals + camarilla_range * 4
    s3_level = close_4h_vals - camarilla_range * 4
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3_level)
    
    # Volume spike filter: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Session filter: 08:00 to 20:00 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need enough data for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators) or outside session
        if (np.isnan(ema34_4h_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or
            np.isnan(volume_spike[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34 = ema34_4h_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Close breaks above R3 + 4h uptrend + volume spike
            if close[i] > r3 and close[i] > ema34 and vol_spike:
                signals[i] = 0.20
                position = 1
            # Enter short: Close breaks below S3 + 4h downtrend + volume spike
            elif close[i] < s3 and close[i] < ema34 and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Close falls below S3 or 4h trend turns down
            if close[i] < s3 or close[i] < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Close rises above R3 or 4h trend turns up
            if close[i] > r3 or close[i] > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals