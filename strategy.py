# 1d_TRIX_ZeroCross_1wTrend_Volume_Spike
# Hypothesis: TRIX zero-cross with 1-week EMA50 trend filter and volume spike.
# TRIX captures momentum changes; zero-cross signals trend shifts. Weekly EMA50 filters for primary trend direction.
# Volume spike confirms institutional participation. Designed for 1d timeframe to reduce trade frequency.
# Works in bull markets (rides trends) and bear markets (catches reversals via zero-cross).
# Target: 15-25 trades/year (60-100 total over 4 years).

name = "1d_TRIX_ZeroCross_1wTrend_Volume_Spike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d TRIX (15-period) ===
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - 1-period percent change
    ema1 = pd.Series(close).ewm(span=15, min_periods=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, min_periods=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, min_periods=15, adjust=False).mean()
    trix = ema3.pct_change() * 100  # Convert to percentage
    trix_values = trix.values
    trix_prev = np.roll(trix_values, 1)
    trix_prev[0] = np.nan
    
    # === 1w EMA50 Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Volume Spike (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5  # 1.5x average volume
    
    # Signal parameters
    position_size = 0.25  # 25% of capital
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (TRIX needs 45 bars for triple EMA)
    start_idx = 45
    
    for i in range(start_idx, n):
        # Skip if any data invalid
        if (np.isnan(trix_values[i]) or np.isnan(trix_prev[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero + above weekly EMA50 + volume spike
            if (trix_prev[i] <= 0 and trix_values[i] > 0 and 
                close[i] > ema50_1w_aligned[i] and volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: TRIX crosses below zero + below weekly EMA50 + volume spike
            elif (trix_prev[i] >= 0 and trix_values[i] < 0 and 
                  close[i] < ema50_1w_aligned[i] and volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: TRIX crosses back through zero (mean reversion of momentum)
            if position == 1 and trix_values[i] < 0:
                signals[i] = 0.0
                position = 0
            elif position == -1 and trix_values[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = position_size if position == 1 else -position_size
    
    return signals

#!/usr/bin/env python3