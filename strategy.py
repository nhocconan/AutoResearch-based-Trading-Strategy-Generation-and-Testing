#!/usr/bin/env python3
# 1D_TRIX_VolumeSpike_1wTrend
# Hypothesis: TRIX (triple exponential average) detects momentum changes, confirmed by volume spikes and weekly trend direction.
# Works in bull markets by capturing momentum continuations, and in bear markets by identifying oversold bounces.
# Uses weekly EMA trend filter to avoid counter-trend trades, reducing whipsaws.
# Target: 20-40 trades/year on 1d timeframe for low friction and high signal quality.

name = "1D_TRIX_VolumeSpike_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate TRIX (15-period) on daily closes
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3.pct_change())
    trix_values = trix.values
    
    # Volume filter: volume > 2.0x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for weekly EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_1w_aligned[i]) or np.isnan(trix_values[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below weekly EMA50
        price_above_ema = close[i] > ema_1w_aligned[i]
        price_below_ema = close[i] < ema_1w_aligned[i]
        
        if position == 0:
            # Long entry: TRIX turns positive + price above weekly EMA + volume spike
            if (trix_values[i] > 0 and 
                trix_values[i-1] <= 0 and  # TRIX just crossed above zero
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX turns negative + price below weekly EMA + volume spike
            elif (trix_values[i] < 0 and 
                  trix_values[i-1] >= 0 and  # TRIX just crossed below zero
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX turns negative or volume drops below average
            if (trix_values[i] < 0 and trix_values[i-1] >= 0) or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX turns positive or volume drops below average
            if (trix_values[i] > 0 and trix_values[i-1] <= 0) or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals