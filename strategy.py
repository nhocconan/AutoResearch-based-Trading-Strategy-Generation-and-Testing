#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Volume Spike with 1d Trend Filter.
# Uses ADX > 20 to confirm trending markets and Volume Spike > 2x average for momentum confirmation.
# Direction from 1d EMA50 slope (rising/falling). Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend).
# Target: 20-40 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on daily close
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ADX(14) on 6h data
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(high[i] - high[i-1], 0)
        minus_dm[i] = max(low[i-1] - low[i], 0)
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        smoothed = np.full_like(data, np.nan)
        if len(data) < period:
            return smoothed
        smoothed[period-1] = np.mean(data[1:period])
        for i in range(period, len(data)):
            smoothed[i] = (smoothed[i-1] * (period-1) + data[i]) / period
        return smoothed
    
    tr14 = wilder_smooth(tr, 14)
    plus_dm14 = wilder_smooth(plus_dm, 14)
    minus_dm14 = wilder_smooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di14 = np.where(tr14 != 0, 100 * plus_dm14 / tr14, 0)
    minus_di14 = np.where(tr14 != 0, 100 * minus_dm14 / tr14, 0)
    dx = np.where((plus_di14 + minus_di14) != 0, 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilder_smooth(dx, 14)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for ADX and volume MA
    start_idx = max(50, 100)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(adx[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from daily EMA50 slope
        if i > 0 and not np.isnan(ema_50_1d_aligned[i-1]):
            trend_up = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            trend_down = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]
        else:
            trend_up = False
            trend_down = False
        
        if position == 0:
            # Long entry: ADX > 20 (trending) + uptrend + volume spike
            if (adx[i] > 20 and 
                trend_up and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: ADX > 20 (trending) + downtrend + volume spike
            elif (adx[i] > 20 and 
                  trend_down and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: ADX < 20 (no trend) or trend turns down
            if (adx[i] < 20 or 
                not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: ADX < 20 (no trend) or trend turns up
            if (adx[i] < 20 or 
                not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX20_VolumeSpike_1dEMA50_v1"
timeframe = "6h"
leverage = 1.0