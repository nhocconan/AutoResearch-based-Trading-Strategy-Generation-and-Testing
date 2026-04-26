#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: On 4h timeframe, price breaking above Camarilla R3 or below S3 with 1d trend filter (price > 1d EMA34) and volume spike (>1.5x 20-period mean) captures strong momentum moves. Uses discrete sizing (±0.30) and close-based stops to target 20-50 trades/year. Works in both bull/bear markets by only trading in direction of higher-timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for higher-timeframe trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Camarilla levels from previous day (using 1d OHLC)
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We use the previous completed 1d bar's levels
    prev_1d_high = df_1d['high'].shift(1).values  # previous day's high
    prev_1d_low = df_1d['low'].shift(1).values    # previous day's low
    prev_1d_close = df_1d['close'].shift(1).values # previous day's close
    
    # Calculate Camarilla levels for previous day
    prev_range = prev_1d_high - prev_1d_low
    r3 = prev_1d_close + 1.1 * prev_range
    s3 = prev_1d_close - 1.1 * prev_range
    
    # Align Camarilla levels to 4h timeframe (one-day delay for completion)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike filter: current volume > 1.5x 20-period mean
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    
    # Warmup: max of calculations (20 for vol MA, 1d EMA34 alignment)
    start_idx = max(20, 34) + 4  # +4 to ensure 1d bar completion (4h -> 1d: 6 bars per day, but we use alignment)
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from calculation)
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 1d trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_1d = close_val > ema_34_val
        bearish_1d = close_val < ema_34_val
        
        # Entry conditions: price breaks above R3 or below S3 with trend and volume spike
        long_entry = (close_val > r3_val) and bullish_1d and vol_spike
        short_entry = (close_val < s3_val) and bearish_1d and vol_spike
        
        # Exit conditions: price returns inside Camarilla H-L range or trend reversal
        # We exit when price moves back towards the mean (between R3 and S3) or trend changes
        long_exit = (close_val < r3_val) or (close_val > s3_val) or not bullish_1d
        short_exit = (close_val > s3_val) or (close_val < r3_val) or not bearish_1d
        
        # Simplified exit: flip signal on opposite condition or mean reversion
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and (close_val < r3_val or not bullish_1d):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close_val > s3_val or not bearish_1d):
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0