#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h EMA Pullback with 4h Trend and 1d Volume Confirmation
# Hypothesis: In trending markets (4h EMA21), price pulls back to 1h EMA50 during strong volume (1d average), offering high-probability entries.
# Works in bull/bear by following 4h trend. Target: 20-40 trades/year (80-160 total) via strict 3-condition confluence.

name = "1h_ema_pullback_4h_trend_1d_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # 1h EMA50 for entry
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 4h EMA21 for trend
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1d average volume (20-period)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=10).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50[i]) or np.isnan(ema_21_4h_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 1h volume > 1.5x 1d average volume
        vol_condition = volume[i] > (1.5 * vol_avg_1d_aligned[i])
        
        if position == 1:  # Long position
            # Exit: price closes below EMA50 or trend changes to down
            if close[i] < ema_50[i] or close[i] < ema_21_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: price closes above EMA50 or trend changes to up
            if close[i] > ema_50[i] or close[i] > ema_21_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            if vol_condition:
                # Pullback to EMA50 in uptrend: buy near support
                if close[i] >= ema_50[i] * 0.998 and close[i] <= ema_50[i] * 1.002 and close[i] > ema_21_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Pullback to EMA50 in downtrend: sell near resistance
                elif close[i] >= ema_50[i] * 0.998 and close[i] <= ema_50[i] * 1.002 and close[i] < ema_21_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals