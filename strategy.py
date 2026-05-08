#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Daily close for Camarilla calculation
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R1, S1) from previous day's range
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # first value
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Calculate R1 and S1
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Daily trend filter: EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d = (close_1d > ema34_1d).astype(float)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Daily volume spike: current volume > 1.5 * 20-day average
    volume_1d = df_1d['volume'].values
    vol_ma20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma20d * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1 with volume spike and daily uptrend
            long_cond = (close[i] > R1_aligned[i] and vol_spike_aligned[i] and trend_1d_aligned[i] > 0.5)
            
            # Short entry: price breaks below S1 with volume spike and daily downtrend
            short_cond = (close[i] < S1_aligned[i] and vol_spike_aligned[i] and trend_1d_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below S1 (mean reversion to support)
            if close[i] < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above R1 (mean reversion to resistance)
            if close[i] > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 breakout on 4H with volume confirmation and daily trend filter.
# Works in bull markets (breakouts continue) and bear markets (mean reversion at opposite level).
# Daily EMA34 ensures alignment with longer-term trend, reducing counter-trend trades.
# Volume spike filter (1.5x 20-day average) ensures momentum confirmation.
# Target: 20-40 trades/year to minimize fee decay while capturing significant moves.