#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_12hEMA34_Trend_VolumeConfirm
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume confirmation (>1.5x 20-bar avg).
Enters long when price breaks above R3 in 12h uptrend, short when breaks below S3 in 12h downtrend.
Uses discrete sizing (0.25) to limit fee churn. Designed for 4h timeframe with ~20-50 trades/year,
works in bull/bear by following 12h trend filter.
"""

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
    open_price = prices['open'].values
    
    # 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Camarilla levels from previous day
    # Camarilla: based on previous day's high, low, close
    # We need daily data
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels for current day based on previous day's OHLC
    # R4 = close + (high-low)*1.1/2
    # R3 = close + (high-low)*1.1/4
    # S3 = close - (high-low)*1.1/4
    # S4 = close - (high-low)*1.1/2
    # We shift by 1 to use previous day's values
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev[0] = high_1d[0]  # fill first value
    low_1d_prev[0] = low_1d[0]
    close_1d_prev[0] = close_1d[0]
    
    # Calculate Camarilla levels
    camarilla_range = (high_1d_prev - low_1d_prev) * 1.1
    r3 = close_1d_prev + camarilla_range / 4
    s3 = close_1d_prev - camarilla_range / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough data for calculations
    start_idx = max(34, 20)  # EMA34 and vol MA20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 in 12h uptrend with volume confirmation
            bullish_setup = (close[i] > r3_aligned[i]) and (close_12h[i] > ema_34_12h_aligned[i]) and volume_spike[i]
            # Short: price breaks below S3 in 12h downtrend with volume confirmation
            bearish_setup = (close[i] < s3_aligned[i]) and (close_12h[i] < ema_34_12h_aligned[i]) and volume_spike[i]
            
            if bullish_setup:
                signals[i] = 0.25
                position = 1
            elif bearish_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below S3 OR trend turns down
            if (close[i] < s3_aligned[i]) or (close_12h[i] < ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above R3 OR trend turns up
            if (close[i] > r3_aligned[i]) or (close_12h[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA34_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0