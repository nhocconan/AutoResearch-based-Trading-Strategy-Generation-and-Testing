# 6h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R3/S3 levels act as strong intraday support/resistance. Breakouts with volume and daily trend filter capture momentum moves.
# Works in bull/bear: In bull markets, breaks above R3 continue up; in bear markets, breaks below S3 continue down. Volume confirms institutional participation.
# Uses 1d trend filter to avoid counter-trend trades. Targets 50-150 trades over 4 years (12-37/year) with disciplined entry.
# Timeframe: 6h allows capturing multi-day moves while reducing noise vs lower timeframes.

#!/usr/bin/env python3
name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Load 1d data once for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Daily OHLC for Camarilla calculations (previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3 for previous day
    # R3/H3 = Close + (High - Low) * 1.1/4
    # S3/L3 = Close - (High - Low) * 1.1/4
    diff = high_1d - low_1d
    r3 = close_1d_vals + diff * 1.1 / 4
    s3 = close_1d_vals - diff * 1.1 / 4
    
    # Align Camarilla levels to 6h (wait for daily close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d_vals).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + above 1d EMA34 + volume spike
            if (close[i] > r3_aligned[i] and close[i] > ema_34_1d_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + below 1d EMA34 + volume spike
            elif (close[i] < s3_aligned[i] and close[i] < ema_34_1d_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below S3 (re-test of broken level as support)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above R3 (re-test of broken level as resistance)
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals