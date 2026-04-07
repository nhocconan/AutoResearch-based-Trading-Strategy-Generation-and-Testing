#!/usr/bin/env python3
"""
6h Camarilla Pivot + Volume + 1d EMA Trend
Long when price breaks above S3 with volume and 1d EMA up
Short when price breaks below R3 with volume and 1d EMA down
Exit when price crosses back through S3/R3 or EMA trend fails
Camarilla levels work well in ranging markets while EMA filter adapts to trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_volume_ema_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d EMA for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Volume ratio ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # Daily pivot points (using previous day's OHLC)
        if i >= 4:  # Need at least 4 periods back for 6h data (1 day = 4 periods)
            # Get previous day's OHLC (4 periods back)
            prev_high = high[i-4]
            prev_low = low[i-4]
            prev_close = close[i-4]
            
            # Calculate Camarilla levels
            range_val = prev_high - prev_low
            if range_val <= 0:
                signals[i] = 0.0
                continue
                
            # Camarilla levels
            s3 = prev_close - (range_val * 1.1000 / 6)
            s4 = prev_close - (range_val * 1.5000 / 6)
            r3 = prev_close + (range_val * 1.1000 / 6)
            r4 = prev_close + (range_val * 1.5000 / 6)
            
            if position == 1:  # Long position
                # Exit: price crosses below S3 or EMA trend turns down
                if close[i] < s3 or ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
            elif position == -1:  # Short position
                # Exit: price crosses above R3 or EMA trend turns up
                if close[i] > r3 or ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Flat, look for entry
                # Need expanding volume
                if vol_ratio[i] < 1.5:
                    signals[i] = 0.0
                    continue
                
                # Entry: Camarilla breakout with volume and trend
                if close[i] > s3 and close[i-1] <= s3 and ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                    # Break above S3 with rising EMA -> long
                    position = 1
                    signals[i] = 0.25
                elif close[i] < r3 and close[i-1] >= r3 and ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                    # Break below R3 with falling EMA -> short
                    position = -1
                    signals[i] = -0.25
    
    return signals