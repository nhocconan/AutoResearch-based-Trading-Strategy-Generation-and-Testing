#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels (R3/S3) from daily charts act as strong support/resistance.
# Breakout above R3 or below S3 with volume confirmation and daily trend filter (price > EMA34) captures strong moves.
# Designed for low trade frequency (20-40/year) to minimize fee fracturing in ranging and trending markets.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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
    
    # Daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate Camarilla levels (R3, S3) from prior day
    # Typical price = (H+L+C)/3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_hl = df_1d['high'] - df_1d['low']
    # Camarilla R3 = Close + (High-Low)*1.1/2
    # Camarilla S3 = Close - (High-Low)*1.1/2
    r3 = df_1d['close'] + range_hl * 1.1 / 2
    s3 = df_1d['close'] - range_hl * 1.1 / 2
    
    # Daily trend filter: EMA34
    close_1d = df_1d['close']
    ema_34 = close_1d.ewm(span=34, adjust=False, min_periods=34).mean()
    
    # Align to 4h timeframe (wait for daily close)
    r3_4h = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3.values)
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34.values)
    
    # Volume confirmation (20-period average on 4h)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 35) + 5
    
    for i in range(start_idx, n):
        if np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or np.isnan(ema_34_4h[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: break above R3 with daily uptrend and volume
            if close[i] > r3_4h[i] and close[i] > ema_34_4h[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with daily downtrend and volume
            elif close[i] < s3_4h[i] and close[i] < ema_34_4h[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below EMA34 or breaks below S3 (reversal)
            if close[i] < ema_34_4h[i] or close[i] < s3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above EMA34 or breaks above R3 (reversal)
            if close[i] > ema_34_4h[i] or close[i] > r3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals