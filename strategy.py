#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Pivot_Reversal_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once for trend filter and pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 15:
        return np.zeros(n)
    
    # 12h close for trend filter: EMA21
    close_12h = df_12h['close'].values
    ema21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    trend_12h = (close_12h > ema21_12h).astype(float)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # 12h volume for spike filter: current volume > 1.5 * 10-period average
    volume_12h = df_12h['volume'].values
    vol_ma10_12h = pd.Series(volume_12h).rolling(window=10, min_periods=10).mean().values
    vol_spike_12h = volume_12h > (vol_ma10_12h * 1.5)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # Calculate daily pivot points (standard floor trader pivots) from previous day
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Pivot point: P = (H + L + C) / 3
    # Support 1: S1 = 2*P - H
    # Resistance 1: R1 = 2*P - L
    P = (prev_high + prev_low + prev_close) / 3.0
    S1 = 2 * P - prev_high
    R1 = 2 * P - prev_low
    
    # Align pivot levels to 6h timeframe
    P_aligned = align_htf_to_ltf(prices, df_1d, P)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(P_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(trend_12h_aligned[i]) or 
            np.isnan(vol_spike_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1 with 12h uptrend and volume spike
            long_cond = (close[i] > R1_aligned[i] and 
                        trend_12h_aligned[i] > 0.5 and 
                        vol_spike_12h_aligned[i])
            
            # Short entry: price breaks below S1 with 12h downtrend and volume spike
            short_cond = (close[i] < S1_aligned[i] and 
                         trend_12h_aligned[i] < 0.5 and 
                         vol_spike_12h_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below pivot (mean reversion to pivot)
            if close[i] < P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above pivot (mean reversion to pivot)
            if close[i] > P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h pivot reversal with 12h trend filter and volume confirmation.
# Uses daily pivot points (P, S1, R1) from previous day for mean reversion entries.
# Long when price breaks above R1 with 12h uptrend and volume spike.
# Short when price breaks below S1 with 12h downtrend and volume spike.
# Exit when price returns to pivot point (P).
# Designed to work in both bull (breakout continuation) and bear (mean reversion) markets.
# Target: 15-25 trades/year to minimize fee decay while capturing significant moves.