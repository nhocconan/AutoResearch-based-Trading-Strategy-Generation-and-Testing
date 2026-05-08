#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_Breakout_12hTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h = (close_12h > ema50_12h).astype(float)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Calculate Camarilla levels from previous day
    # Previous day's high, low, close
    prev_day_high = np.maximum.accumulate(high)
    prev_day_low = np.minimum.accumulate(low)
    prev_day_close = np.concatenate([[close[0]], close[:-1]])
    
    # Shift by one day to get previous day's values
    prev_high = np.concatenate([[prev_day_high[0]], prev_day_high[:-1]])
    prev_low = np.concatenate([[prev_day_low[0]], prev_day_low[:-1]])
    prev_close = np.concatenate([[prev_day_close[0]], prev_day_close[:-1]])
    
    # Camarilla levels: R3, S3
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 with volume spike and 12h uptrend
            long_cond = (close[i] > R3[i] and vol_spike[i] and trend_12h_aligned[i] > 0.5)
            
            # Short entry: price breaks below S3 with volume spike and 12h downtrend
            short_cond = (close[i] < S3[i] and vol_spike[i] and trend_12h_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.30
                position = 1
            elif short_cond:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price closes below S3 (mean reversion)
            if close[i] < S3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price closes above R3 (mean reversion)
            if close[i] > R3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout with volume confirmation and 12h trend filter on 4h timeframe.
# Works in bull markets (breakouts continue) and bear markets (mean reversion at opposite level).
# Uses 12h EMA50 for trend filter to avoid counter-trend trades.
# Volume spike requirement (2x 20-period average) reduces false breakouts.
# Target: 20-40 trades/year to minimize fee decay while capturing significant moves.