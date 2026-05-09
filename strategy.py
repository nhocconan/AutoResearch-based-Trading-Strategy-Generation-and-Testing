# 6h_Camarilla_R3S3_Fade_1dTrend_Confirm
# Strategy type: Counter-trend fade at Camarilla R3/S3 with 1d trend confirmation
# Rationale: Camarilla R3/S3 act as strong reversal zones; 1d trend filters direction to avoid counter-trend trades in strong trends; volume confirms rejection.
# Works in bull/bear by fading extremes only when higher timeframe trend is overextended.
# Target: 20-40 trades/year per symbol with strict entry conditions.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R3S3_Fade_1dTrend_Confirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Using standard formula based on previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels
    # R4 = Close + (High - Low) * 1.5
    # R3 = Close + (High - Low) * 1.25
    # S3 = Close - (High - Low) * 1.25
    # S4 = Close - (High - Low) * 1.5
    r4 = prev_close + (prev_high - prev_low) * 1.5
    r3 = prev_close + (prev_high - prev_low) * 1.25
    s3 = prev_close - (prev_high - prev_low) * 1.25
    s4 = prev_close - (prev_high - prev_low) * 1.5
    
    # Align daily Camarilla levels to 6h timeframe (with 1-bar delay for completed day)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: spike above 2x 24-period average (more stringent for 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Wait for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_6h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation spike
        
        # Pre-compute hour for session filter (UTC 0-24)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Trade during active hours: 8 AM - 8 PM UTC
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Fade long at S3: price below S3, 1d uptrend (price > EMA50), volume spike
            if (close[i] < s3_6h[i] and 
                close[i] > ema_50_6h[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Fade short at R3: price above R3, 1d downtrend (price < EMA50), volume spike
            elif (close[i] > r3_6h[i] and 
                  close[i] < ema_50_6h[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses above S4 (stop) or reaches R3 (target)
            if close[i] > s4_6h[i] or close[i] > r3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below S4 (stop) or reaches R3 (target)
            if close[i] < s4_6h[i] or close[i] < s3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals