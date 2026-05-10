#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1wEMA34_Trend_Volume
Hypothesis: 12h timeframe with weekly EMA34 trend filter and weekly RSI filter for regime.
Combines Camarilla pivot breakouts (R3/S3) with weekly trend and momentum filters.
Designed to work in both bull and bear markets by using weekly EMA for trend direction
and weekly RSI to avoid overextended moves. Target: 15-25 trades/year to minimize fee drag.
"""

name = "12h_Camarilla_R3S3_Breakout_1wEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend and momentum filters
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly RSI(14) for momentum filter
    close_1w = pd.Series(df_1w['close'].values)
    delta = close_1w.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1w = 100 - (100 / (1 + rs))
    rsi_14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w.values)
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = df_1d['high'].values
    low_prev = df_1d['low'].values
    close_prev = df_1d['close'].values
    
    # Camarilla formulas
    camarilla_r3 = close_prev + (high_prev - low_prev) * 1.25
    camarilla_s3 = close_prev - (high_prev - low_prev) * 1.25
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get price, volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2.0x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA34 (34) and RSI (14)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(rsi_14_1w_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 with uptrend, not overbought, and volume
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                rsi_14_1w_aligned[i] < 70 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with downtrend, not oversold, and volume
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  rsi_14_1w_aligned[i] > 30 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price back below S3 or trend change or overbought
            if (close[i] < camarilla_s3_aligned[i] or 
                close[i] < ema_34_1w_aligned[i] or 
                rsi_14_1w_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price back above R3 or trend change or oversold
            if (close[i] > camarilla_r3_aligned[i] or 
                close[i] > ema_34_1w_aligned[i] or 
                rsi_14_1w_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals