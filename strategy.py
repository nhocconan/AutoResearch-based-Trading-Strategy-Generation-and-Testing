#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation.
- Camarilla pivots calculated from previous 4h bar (HTF structure)
- Long: price breaks above R3 with volume > 2x 20-period average AND price > 4h EMA50
- Short: price breaks below S3 with volume > 2x 20-period average AND price < 4h EMA50
- Exit: price returns to 4h EMA50 or opposite Camarilla breakout
- Uses 4h for signal direction/trend filter, 1h only for entry timing precision
- Session filter: 08-20 UTC to avoid low-liquidity hours
- Discrete position sizing: ±0.20 to minimize fee churn
- Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe
- Works in bull markets (breakouts with trend) and bear markets (mean reversion from extremes)
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 4h bar
    # Typical price = (high + low + close) / 3
    typical_price_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    typical_price_4h_vals = typical_price_4h.values
    
    # Camarilla width = (high - low) * 1.1 / 12
    camarilla_width_4h = (df_4h['high'] - df_4h['low']) * 1.1 / 12
    camarilla_width_4h_vals = camarilla_width_4h.values
    
    # R3 = close + (high - low) * 1.1 * 1.125/4
    # S3 = close - (high - low) * 1.1 * 1.125/4
    # Using standard Camarilla multipliers: R3/S3 = close ± 1.1 * (high-low) * 1.125/4
    camarilla_multiplier = 1.1 * 1.125 / 4
    r3_4h = df_4h['close'] + (df_4h['high'] - df_4h['low']) * camarilla_multiplier
    s3_4h = df_4h['close'] - (df_4h['high'] - df_4h['low']) * camarilla_multiplier
    r3_4h_vals = r3_4h.values
    s3_4h_vals = s3_4h.values
    
    # Align Camarilla levels to 1h timeframe (previous 4h bar's levels)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h_vals)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h_vals)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(r3_4h_aligned[i]) or
            np.isnan(s3_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 + volume confirmation + price > 4h EMA50
            if (close[i] > r3_4h_aligned[i] and 
                volume_confirm and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 + volume confirmation + price < 4h EMA50
            elif (close[i] < s3_4h_aligned[i] and 
                  volume_confirm and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price returns to 4h EMA50 OR price breaks below S3 (mean reversion)
            if (close[i] <= ema_50_4h_aligned[i] or 
                close[i] < s3_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price returns to 4h EMA50 OR price breaks above R3 (mean reversion)
            if (close[i] >= ema_50_4h_aligned[i] or 
                close[i] > r3_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0