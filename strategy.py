#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA(50) trend filter and volume confirmation
# Long when price breaks above Camarilla R3 + volume spike + price > 12h EMA(50)
# Short when price breaks below Camarilla S3 + volume spike + price < 12h EMA(50)
# Uses Camarilla levels from previous 4h bar to avoid look-ahead
# 12h EMA(50) filter captures intermediate trend and reduces whipsaw
# Volume spike (2.0x 20-period average) confirms institutional participation
# Designed for low trade frequency (19-50/year on 4h) to minimize fee drag
# Works in both bull (breakouts) and bear (mean reversion at extremes) markets

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA(50) trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 4h timeframe (wait for completed 12h bar)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels from previous 4h bar
    # Based on previous bar's high, low, close
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous bar's OHLC to calculate today's Camarilla levels
        high_prev = high[i-1]
        low_prev = low[i-1]
        close_prev = close[i-1]
        
        # Calculate pivot point
        pivot = (high_prev + low_prev + close_prev) / 3.0
        
        # Calculate range
        range_ = high_prev - low_prev
        
        # Camarilla levels
        camarilla_r3[i] = close_prev + (range_ * 1.1 / 4.0)
        camarilla_s3[i] = close_prev - (range_ * 1.1 / 4.0)
    
    # Volume confirmation (2.0x 20-period average) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 50  # max(1 for Camarilla, 20 for volume MA, 50 for 12h EMA)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 + volume spike + price > 12h EMA(50)
            if (close[i] > camarilla_r3[i] and volume_spike[i] and close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Camarilla S3 + volume spike + price < 12h EMA(50)
            elif (close[i] < camarilla_s3[i] and volume_spike[i] and close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla S3 OR price below 12h EMA(50)
            if (close[i] < camarilla_s3[i] or close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R3 OR price above 12h EMA(50)
            if (close[i] > camarilla_r3[i] or close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals