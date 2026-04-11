#!/usr/bin/env python3
"""
1d_1w_camarilla_breakout_volume_v1
Strategy: Daily Camarilla Pivot breakout with volume confirmation and weekly trend filter
Timeframe: 1d
Leverage: 1.0
Hypothesis: Uses daily Camarilla Pivot levels (resistance/support) for breakout entries confirmed by volume spike (>1.5x average volume) and filtered by weekly EMA50 trend direction. Designed to capture strong momentum moves in trending markets while avoiding false breakouts in chop. Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered out). Target: 30-100 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close arrays"""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    R4 = close + range_val * 1.5000
    R3 = close + range_val * 1.2500
    R2 = close + range_val * 1.1666
    R1 = close + range_val * 1.0833
    S1 = close - range_val * 1.0833
    S2 = close - range_val * 1.1666
    S3 = close - range_val * 1.2500
    S4 = close - range_val * 1.5000
    return R1, R2, R3, R4, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    R1, R2, R3, R4, S1, S2, S3, S4 = calculate_camarilla(high, low, close)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(R1[i]) or np.isnan(R4[i]) or 
            np.isnan(S1[i]) or np.isnan(S4[i]) or
            np.isnan(vol_avg[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below weekly EMA50
        uptrend_1w = price_close > ema_50_1w_aligned[i]
        downtrend_1w = price_close < ema_50_1w_aligned[i]
        
        # Breakout conditions (using previous day's levels to avoid look-ahead)
        breakout_up = price_close > R4[i-1]  # Break above R4
        breakout_down = price_close < S4[i-1]  # Break below S4
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: upward breakout with volume in uptrend
        long_signal = breakout_up and vol_confirmed and uptrend_1w
        
        # Short: downward breakout with volume in downtrend
        short_signal = breakout_down and vol_confirmed and downtrend_1w
        
        # Exit when price returns to R3/S3 levels (profit taking)
        exit_long = position == 1 and price_close < R3[i]
        exit_short = position == -1 and price_close > S3[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals