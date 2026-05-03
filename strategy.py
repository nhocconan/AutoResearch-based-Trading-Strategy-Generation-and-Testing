#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long: Close breaks above R3 AND price > 1d EMA34 (uptrend) AND volume > 2.0x 20-period MA
# Short: Close breaks below S3 AND price < 1d EMA34 (downtrend) AND volume > 2.0x 20-period MA
# Exit: Opposite pivot breakout or EMA34 trend reversal.
# Discrete sizing 0.20. Target: 60-150 total trades over 4 years (15-37/year).
# Uses 4h/1d for signal direction, 1h only for entry timing. Session filter 08-20 UTC reduces noise.

name = "1h_Camarilla_R3S3_1dEMA34_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 arithmetic)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 4h (using previous bar's H/L/C)
    prev_high_4h = np.concatenate([[np.nan], df_4h['high'].values[:-1]])
    prev_low_4h = np.concatenate([[np.nan], df_4h['low'].values[:-1]])
    prev_close_4h = np.concatenate([[np.nan], df_4h['close'].values[:-1]])
    
    camarilla_range_4h = prev_high_4h - prev_low_4h
    r3_4h = prev_close_4h + (camarilla_range_4h * 1.2500)
    s3_4h = prev_close_4h - (camarilla_range_4h * 1.2500)
    
    # Align 4h Camarilla levels to 1h (waits for completed 4h bar)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume regime: current 1h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if any value is NaN
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        r3 = r3_4h_aligned[i]
        s3 = s3_4h_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: Close breaks above R3 AND uptrend AND volume spike
            if close_val > r3 and is_uptrend and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: Close breaks below S3 AND downtrend AND volume spike
            elif close_val < s3 and is_downtrend and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Close breaks below S3 OR trend turns down
            if close_val < s3 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Close breaks above R3 OR trend turns up
            if close_val > r3 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals