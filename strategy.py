#148600
#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
Camarilla levels identify intraday support/resistance. Breakouts above R3 or below S3
with trend alignment and volume confirmation capture strong momentum moves.
Works in bull (breakouts above R3) and bear (breakouts below S3).
Target: 75-200 total trades over 4 years (19-50/year).
"""

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema34_1d[i-1]
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R3 = close + 1.1*(high-low)/1.1, S3 = close - 1.1*(high-low)/1.1
    # Actually: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #          S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        if not np.isnan(high[i]) and not np.isnan(low[i]) and not np.isnan(close[i-1]):
            prev_high = high[i-1]  # previous day's high
            prev_low = low[i-1]    # previous day's low
            prev_close = close[i-1] # previous day's close
            camarilla_r3[i] = prev_close + 1.1 * (prev_high - prev_low)
            camarilla_s3[i] = prev_close - 1.1 * (prev_high - prev_low)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 1)  # warmup for EMA and Camarilla
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or \
           np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 4h volume from 1d
        vol_4h_approx = vol_sma20_1d_aligned[i] / 6.0
        volume_confirm = volume[i] > 1.5 * vol_4h_approx
        
        if position == 0:
            # Long: Price breaks above R3 with uptrend and volume confirmation
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with downtrend and volume confirmation
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price re-enters below R3 or trend reversal
            if close[i] < camarilla_r3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price re-enters above S3 or trend reversal
            if close[i] > camarilla_s3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals