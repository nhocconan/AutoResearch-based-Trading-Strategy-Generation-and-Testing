#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation.
# Uses 1h primary timeframe targeting 15-37 trades/year (60-150 total over 4 years).
# Camarilla levels from 4h provide intraday structure, 4h EMA50 filters trend direction,
# and volume spike confirms momentum. Session filter (08-20 UTC) reduces noise trades.
# Designed for BTC/ETH to work in both bull and bear markets by taking breakouts
# in the direction of the higher timeframe trend.

name = "1h_Camarilla_R1S1_4hEMA50_VolumeSpike_Session"
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
    open_time = prices['open_time'].values
    
    # Get 4h data for Camarilla pivots and EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 trend filter
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate Camarilla pivot levels from previous 4h bar
    # Typical price = (H + L + C) / 3
    typical_price = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    typical_price_values = typical_price.values
    
    # Camarilla levels: R1 = PP + (H - L) * 1.1/12, S1 = PP - (H - L) * 1.1/12
    # Using previous 4h bar's values (already completed bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_vals = df_4h['close'].values
    
    # Pivot point
    pp = (high_4h + low_4h + close_4h_vals) / 3
    # Range
    rng = high_4h - low_4h
    # Camarilla levels
    r1 = pp + (rng * 1.1 / 12)
    s1 = pp - (rng * 1.1 / 12)
    
    # Align Camarilla levels to 1h timeframe (wait for 4h bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Calculate volume regime: current 1h volume > 2.0x 20-period MA (strict to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if outside session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Get current values
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Entry conditions
        # Long: break above R1 with volume spike and above 4h EMA50
        long_entry = (close[i] > r1_val) and vol_spike and (close[i] > ema_trend)
        # Short: break below S1 with volume spike and below 4h EMA50
        short_entry = (close[i] < s1_val) and vol_spike and (close[i] < ema_trend)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit on close below 4h EMA50 (trend change)
            if close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit on close above 4h EMA50 (trend change)
            if close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals