#!/usr/bin/env python3
# 1h_4h1dTrend_WithVolumeSpike_Entry
# Hypothesis: Trade in direction of 4h and 1d trend alignment using EMA crossovers.
# Enter on 1h pullbacks with volume spike confirmation during active hours (08-20 UTC).
# Uses 4h EMA20/50 and 1d EMA50 for trend filter, 1h RSI for entry timing.
# Designed to work in both bull and bear markets by following higher timeframe trends.
# Targets ~20-30 trades/year to minimize fee drag.

name = "1h_4h1dTrend_WithVolumeSpike_Entry"
timeframe = "1h"
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
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA20/50 trend
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_4h_up = ema20_4h > ema50_4h
    trend_4h_down = ema20_4h < ema50_4h
    
    # Align 4h trend to 1h
    trend_4h_up_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_up.astype(float))
    trend_4h_down_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_down.astype(float))
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align 1d trend to 1h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # 1h RSI for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1h volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(trend_4h_up_aligned[i]) or np.isnan(trend_4h_down_aligned[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: aligned uptrend on 4h and 1d, RSI oversold, volume spike, in session
            if (trend_4h_up_aligned[i] > 0.5 and
                trend_1d_up_aligned[i] > 0.5 and
                rsi[i] < 30 and
                vol_spike[i] and
                in_session[i]):
                signals[i] = 0.20
                position = 1
            # Short: aligned downtrend on 4h and 1d, RSI overbought, volume spike, in session
            elif (trend_4h_down_aligned[i] > 0.5 and
                  trend_1d_down_aligned[i] > 0.5 and
                  rsi[i] > 70 and
                  vol_spike[i] and
                  in_session[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: RSI overbought or trend breaks down
            if (rsi[i] > 70 or
                trend_4h_up_aligned[i] < 0.5 or
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: RSI oversold or trend breaks up
            if (rsi[i] < 30 or
                trend_4h_down_aligned[i] < 0.5 or
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals