#!/usr/bin/env python3
# 1h_Camarilla_Pivot_4hTrend_1dVolumeFilter
# Hypothesis: Use 4h trend (price above/below EMA20) for direction, 1d volume spike for confirmation, and Camarilla pivot breakout on 1h for entry timing.
# Works in bull/bear by following higher timeframe trend. Volume filter reduces false breakouts. Targets 15-35 trades/year.

name = "1h_Camarilla_Pivot_4hTrend_1dVolumeFilter"
timeframe = "1h"
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
    
    # 4h trend filter: EMA20
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h Camarilla pivot levels (using previous hour's range)
    # Calculate pivot and levels from previous bar's high/low/close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    
    # Camarilla levels (using standard multipliers)
    r3 = pivot + (range_ * 1.1 / 4)
    s3 = pivot - (range_ * 1.1 / 4)
    
    # 1d volume filter: volume > 1.5x 20-period MA
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 4h EMA to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(vol_ma_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > EMA4h (uptrend) + breaks above S3 + volume spike
            if close[i] > ema_4h_aligned[i] and close[i] > s3[i] and volume[i] > vol_ma_1d[i] * 1.5:
                signals[i] = 0.20
                position = 1
            # SHORT: price < EMA4h (downtrend) + breaks below R3 + volume spike
            elif close[i] < ema_4h_aligned[i] and close[i] < r3[i] and volume[i] > vol_ma_1d[i] * 1.5:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below EMA4h (trend change)
            if close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price crosses above EMA4h (trend change)
            if close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals