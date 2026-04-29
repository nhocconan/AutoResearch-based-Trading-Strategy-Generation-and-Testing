#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike
# Uses Camarilla pivot levels (R3/S3) for high-probability breakout entries
# 4h EMA50 provides trend filter to avoid counter-trend trades in both bull/bear markets
# Volume > 2.0x average confirms institutional participation and reduces false breakouts
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Discrete position sizing (0.20) with R3/S3 reversion exit for quick profit taking
# Designed for ~15-35 trades/year to minimize fee drag while capturing strong moves
# Works in bull/bear via trend filter - only trades in direction of 4h EMA50

name = "1h_Camarilla_R3S3_4hEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Camarilla pivot points (R3, S3 levels)
    # Typical Price = (H + L + C) / 3
    typical_price = (high + low + close) / 3.0
    # Pivot Point = Typical Price
    pp = typical_price
    # R3 = PP + 2*(High - Low)  [Aggressive resistance]
    # S3 = PP - 2*(High - Low)  [Aggressive support]
    camarilla_r3 = pp + 2.0 * (high - low)
    camarilla_s3 = pp - 2.0 * (high - low)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Volume MA and 4h EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_r3 = camarilla_r3[i]
        curr_s3 = camarilla_s3[i]
        curr_ema50_4h = ema_50_4h_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below S3 (mean reversion to pivot)
            if curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price above R3 (mean reversion to pivot)
            if curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirmed = curr_volume > 2.0 * curr_vol_ma
            
            # Long when price breaks above R3, 4h EMA50 up-trend, volume confirmed, in session
            if curr_high > curr_r3 and curr_close > curr_ema50_4h and vol_confirmed:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below S3, 4h EMA50 down-trend, volume confirmed, in session
            elif curr_low < curr_s3 and curr_close < curr_ema50_4h and vol_confirmed:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals