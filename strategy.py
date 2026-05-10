#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dEMA50_Trend_Volume
# Hypothesis: Daily EMA50 trend filter reduces false breakouts, while tight Camarilla R1/S3
# levels provide high-probability entries with volume confirmation. Designed for low
# trade frequency (20-40/year) to minimize fee drag. Works in bull via breakouts with
# trend, in bear via mean-reversion at extreme levels when trend aligns.

name = "4h_Camarilla_R1_S1_Breakout_1dEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Daily EMA50 for trend
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Get daily data for Camarilla pivot levels
    # Calculate typical price and range from previous day
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_hl = df_1d['high'] - df_1d['low']
    # Camarilla R1 and S1 levels (tighter than R3/S3 for higher quality)
    R1 = typical_price + (range_hl * 1.0833)
    S1 = typical_price - (range_hl * 1.0833)
    # Align daily levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1.values)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1.values)
    
    # Volume confirmation (20-period average on 4h = ~3.3 days)
    vol_ma_period = 20
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, vol_ma_period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50) + 5  # need enough history for calculations
    
    for i in range(start_idx, n):
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or \
           np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.8x average (balanced for signal quality)
        volume_confirm = volume[i] > 1.8 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price breaks above R1 with volume, above daily EMA50 (uptrend)
            if close[i] > R1_aligned[i] and volume_confirm and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume, below daily EMA50 (downtrend)
            elif close[i] < S1_aligned[i] and volume_confirm and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below S1 or breaks below daily EMA50
            if close[i] < S1_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above R1 or breaks above daily EMA50
            if close[i] > R1_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals