#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: On 12h timeframe, Camarilla R3/S3 breakout with 1w trend filter (price above/below 20-period EMA on 1w) and volume confirmation (>2x average) provides robust signals in both bull and bear markets. Uses discrete sizing (0.0, ±0.25) to minimize fee churn. Targets 50-150 trades over 4 years (12-37/year) for optimal 12h frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # need enough for EMA calculation
        return np.zeros(n)
    
    # Calculate Camarilla levels on 12h using previous day's OHLC
    # We'll approximate using rolling 28-period (2*12h) for previous day's range
    period28_high = pd.Series(high).rolling(window=28, min_periods=28).max().values
    period28_low = pd.Series(low).rolling(window=28, min_periods=28).min().values
    period28_close = pd.Series(close).rolling(window=28, min_periods=28).last().values
    
    # Camarilla R3, S3 levels
    # R3 = close + 1.1*(high-low)/2
    # S3 = close - 1.1*(high-low)/2
    camarilla_r3 = period28_close + 1.1 * (period28_high - period28_low) / 2
    camarilla_s3 = period28_close - 1.1 * (period28_high - period28_low) / 2
    
    # Calculate 1w EMA20 for trend filter
    df_1w_close = df_1w['close'].values
    ema_20_1w = pd.Series(df_1w_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume ratio (current / 20-period average) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Camarilla periods (28), EMA (20), volume MA (20)
    start_idx = max(28, 20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_up = close[i] > camarilla_r3[i]
        breakout_down = close[i] < camarilla_s3[i]
        
        # 1w trend filter
        price_above_1w_ema = close[i] > ema_20_1w_aligned[i]
        price_below_1w_ema = close[i] < ema_20_1w_aligned[i]
        
        vol_confirmed = vol_ratio[i] > 2.0  # volume at least 2.0x average
        
        if position == 0:
            # Long: breakout above R3 + price above 1w EMA + volume
            long_signal = breakout_up and price_above_1w_ema and vol_confirmed
            
            # Short: breakout below S3 + price below 1w EMA + volume
            short_signal = breakout_down and price_below_1w_ema and vol_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below S3 (reversion to mean)
            if close[i] < camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above R3 (reversion to mean)
            if close[i] > camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0