#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_ZeroCross_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d = (close_1d > ema34_1d).astype(float)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Daily volume spike: current volume > 1.3 * 20-day average
    volume_1d = df_1d['volume'].values
    vol_ma20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma20d * 1.3)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # TRIX on 4h close: TRIX = EMA(EMA(EMA(close,12),12),12) - previous value
    # Calculate triple EMA
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX = (ema3 - previous ema3) / previous ema3 * 100
    trix_raw = np.zeros_like(close)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    # Smooth TRIX with 8-period EMA for signal line
    trix = pd.Series(trix_raw).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for TRIX (12+12+12+8)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trix[i]) or np.isnan(trend_1d_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: TRIX crosses above zero with volume spike and daily uptrend
            long_cond = (trix[i] > 0 and trix[i-1] <= 0 and vol_spike_aligned[i] and trend_1d_aligned[i] > 0.5)
            
            # Short entry: TRIX crosses below zero with volume spike and daily downtrend
            short_cond = (trix[i] < 0 and trix[i-1] >= 0 and vol_spike_aligned[i] and trend_1d_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero
            if trix[i] < 0 and trix[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero
            if trix[i] > 0 and trix[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX zero-cross with daily trend filter and volume confirmation.
# TRIX captures momentum changes; zero-cross signals trend acceleration.
# Daily EMA34 ensures alignment with longer-term trend to avoid counter-trend trades.
# Volume spike (1.3x 20-day average) confirms momentum behind the move.
# Works in bull (TRIX > 0 with uptrend) and bear (TRIX < 0 with downtrend) markets.
# Target: ~25 trades/year to minimize fee decay while capturing significant momentum shifts.