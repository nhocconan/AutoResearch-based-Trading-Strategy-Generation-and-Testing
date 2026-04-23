#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation, session filter (08-20 UTC)
- Uses Camarilla pivot levels from 1h timeframe for precise entry/exit points
- 4h EMA50 defines higher timeframe trend: only trade breakouts in trend direction
- Volume confirmation (> 1.8x 20-period average) filters false breakouts
- Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
- Designed for 1h timeframe targeting 15-37 trades/year (60-150 over 4 years)
- Works in both bull and bear markets by trading with the 4h trend
- Camarilla R3/S3 levels provide wider breakout bands for fewer, higher-quality signals
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
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1h Camarilla pivot levels (R3, S3)
    camarilla_r1 = close + (high - low) * 1.1 / 12
    camarilla_s1 = close - (high - low) * 1.1 / 12
    camarilla_r3 = close + (high - low) * 1.1 / 4
    camarilla_s3 = close - (high - low) * 1.1 / 4
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 with 4h uptrend and volume
            long_breakout = (close[i] > camarilla_r3[i] and 
                           close[i] > ema_50_4h_aligned[i] and
                           volume[i] > 1.8 * vol_ma[i])
            
            # Short conditions: price breaks below Camarilla S3 with 4h downtrend and volume
            short_breakout = (close[i] < camarilla_s3[i] and 
                            close[i] < ema_50_4h_aligned[i] and
                            volume[i] > 1.8 * vol_ma[i])
            
            if long_breakout:
                signals[i] = 0.20
                position = 1
            elif short_breakout:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions: opposite Camarilla breakout or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Camarilla S3 or 4h trend turns bearish
                if (close[i] < camarilla_s3[i] or 
                    close[i] < ema_50_4h_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above Camarilla R3 or 4h trend turns bullish
                if (close[i] > camarilla_r3[i] or 
                    close[i] > ema_50_4h_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeConfirm_Session"
timeframe = "1h"
leverage = 1.0