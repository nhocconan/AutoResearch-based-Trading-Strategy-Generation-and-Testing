#!/usr/bin/env python3
# 1d_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Use Camarilla R1/S1 breakout on 1d for entries, filtered by 1w EMA trend and volume spikes.
# Camarilla pivot levels provide high-probability reversal/breakout points in range-bound and trending markets.
# The 1w EMA filter ensures alignment with the weekly trend, reducing counter-trend trades.
# Volume confirmation adds conviction to breakouts, filtering out low-liquidity false signals.
# Designed to work in both bull and bear markets by following the higher-timeframe trend.
# Target: 10-25 trades/year to stay within optimal trade frequency for 1d.

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels for each day using previous day's OHLC
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We use previous day's data to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # Handle first value
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    range_prev = prev_high - prev_low
    camarilla_R1 = prev_close + range_prev * 1.1 / 12
    camarilla_S1 = prev_close - range_prev * 1.1 / 12
    
    # 1w EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_R1[i]) or np.isnan(camarilla_S1[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close crosses above Camarilla R1, 1w EMA uptrend, volume confirmation
            if close[i] > camarilla_R1[i] and close[i-1] <= camarilla_R1[i-1] and ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close crosses below Camarilla S1, 1w EMA downtrend, volume confirmation
            elif close[i] < camarilla_S1[i] and close[i-1] >= camarilla_S1[i-1] and ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Close crosses below Camarilla S1 (mean reversion) or trend reversal
            if close[i] < camarilla_S1[i] or ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Close crosses above Camarilla R1 (mean reversion) or trend reversal
            if close[i] > camarilla_R1[i] or ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals