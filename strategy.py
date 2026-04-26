#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Camarilla R3/S3 breakout with 1d trend filter and volume spike captures strong momentum in both bull and bear markets. Uses 4h timeframe to target 20-50 trades/year with discrete sizing (0.30). Volume confirmation reduces false breakouts.
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Typical price = (H+L+C)/3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    typical_price_values = typical_price.values
    
    # Camarilla width = (H - L) * 1.1 / 12
    camarilla_width = ((df_1d['high'] - df_1d['low']) * 1.1 / 12).values
    
    # R3 = C + (H-L) * 1.1/12 * 4
    # S3 = C - (H-L) * 1.1/12 * 4
    r3 = typical_price_values + camarilla_width * 4
    s3 = typical_price_values - camarilla_width * 4
    
    # Align Camarilla levels to 4h timeframe (1d -> 4h)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    
    # Warmup: max of calculations (20 for volume MA, need 1d data)
    start_idx = 20
    
    for i in range(start_idx, n):
        close_val = close[i]
        ema_val = ema_34_1d_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(r3_val) or np.isnan(s3_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price vs 1d EMA34
        uptrend = close_val > ema_val
        downtrend = close_val < ema_val
        
        # Long: price ABOVE R3 with 1d uptrend and volume spike
        long_condition = (close_val > r3_val) and uptrend and vol_spike
        # Short: price BELOW S3 with 1d downtrend and volume spike
        short_condition = (close_val < s3_val) and downtrend and vol_spike
        
        # Exit: price re-enters between R3 and S3
        long_exit = (position == 1 and close_val <= r3_val)
        short_exit = (position == -1 and close_val >= s3_val)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0