#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 in 1d uptrend (close > EMA34).
# Short when price breaks below Camarilla S3 in 1d downtrend (close < EMA34).
# Volume must be > 1.5x 24-period MA to confirm breakout strength.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 50-150 total trades over 4 years.

name = "12h_Camarilla_R3S3_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Camarilla levels on 12h (using previous bar's high/low/close)
    # R3 = C + (H-L) * 1.1/2, S3 = C - (H-L) * 1.1/2
    camarilla_r3 = np.zeros(n)
    camarilla_s3 = np.zeros(n)
    for i in range(1, n):
        h = high[i-1]
        l = low[i-1]
        c = close[i-1]
        camarilla_r3[i] = c + (h - l) * 1.1 / 2
        camarilla_s3[i] = c - (h - l) * 1.1 / 2
    
    # Volume confirmation: current volume > 1.5x 24-period MA
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        vol_spike = volume_spike[i]
        
        # Entry logic
        if position == 0:
            # Long: price breaks above Camarilla R3 AND 1d uptrend AND volume spike
            if close_val > camarilla_r3[i] and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND 1d downtrend AND volume spike
            elif close_val < camarilla_s3[i] and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla S3 OR 1d trend turns down
            if close_val < camarilla_s3[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Camarilla R3 OR 1d trend turns up
            if close_val > camarilla_r3[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals