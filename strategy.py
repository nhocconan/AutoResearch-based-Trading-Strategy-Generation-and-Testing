#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Momentum_Filter_With_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for momentum and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly close for momentum and trend
    close_1w = df_1w['close'].values
    
    # Weekly momentum: 5-period ROC
    roc_5 = np.zeros_like(close_1w)
    roc_5[5:] = (close_1w[5:] - close_1w[:-5]) / close_1w[:-5] * 100
    
    # Weekly trend filter: EMA21
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    trend_1w = (close_1w > ema21_1w).astype(float)
    
    # Daily volume spike: current volume > 2.0 * 20-day average
    vol_ma20d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20d * 2.0)
    
    # Align weekly indicators to daily
    roc_5_aligned = align_htf_to_ltf(prices, df_1w, roc_5)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for weekly indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(roc_5_aligned[i]) or np.isnan(trend_1w_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: positive weekly momentum + weekly uptrend + volume spike
            long_cond = (roc_5_aligned[i] > 2.0 and trend_1w_aligned[i] > 0.5 and vol_spike[i])
            
            # Short entry: negative weekly momentum + weekly downtrend + volume spike
            short_cond = (roc_5_aligned[i] < -2.0 and trend_1w_aligned[i] < 0.5 and vol_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly momentum turns negative or trend breaks
            if roc_5_aligned[i] < 0 or trend_1w_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly momentum turns positive or trend breaks
            if roc_5_aligned[i] > 0 or trend_1w_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily momentum with weekly trend filter and volume spike confirmation.
# Weekly ROC > 2% indicates strong momentum, aligned with weekly EMA21 trend.
# Volume spike (2x 20-day avg) confirms institutional participation.
# Works in bull markets (momentum continuation) and bear markets (mean reversion via exits).
# Target: 15-25 trades/year to minimize fee decay while capturing significant moves.