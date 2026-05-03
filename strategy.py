#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA(50) trend filter and volume confirmation
# Long when price breaks above Camarilla R3 + volume spike + price > 4h EMA(50)
# Short when price breaks below Camarilla S3 + volume spike + price < 4h EMA(50)
# Uses Camarilla pivot levels from 1h OHLC for precise intraday structure
# 4h EMA(50) filter reduces whipsaw in choppy markets while capturing medium-term trend
# Volume spike (2.0x 20-period average) confirms institutional participation
# Session filter (08-20 UTC) reduces noise trades outside active hours
# Designed for low trade frequency (15-37/year on 1h) to minimize fee drag
# Works in both bull (breakouts) and bear (mean reversion at extremes) markets

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_v1"
timeframe = "1h"
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
    
    # Get 4h data for EMA(50) trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA to 1h timeframe (wait for completed 4h bar)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 1h bar
    # R3 = C + (H-L) * 1.1/2
    # S3 = C - (H-L) * 1.1/2
    # Using previous bar's OHLC to avoid look-ahead
    camarilla_r3 = np.zeros(n)
    camarilla_s3 = np.zeros(n)
    
    for i in range(1, n):
        camarilla_r3[i] = close[i-1] + (high[i-1] - low[i-1]) * 1.1 / 2
        camarilla_s3[i] = close[i-1] - (high[i-1] - low[i-1]) * 1.1 / 2
    
    # Volume confirmation (2.0x 20-period average) on 1h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 50  # max(20 for volume MA, 50 for 4h EMA)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside session
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 + volume spike + price > 4h EMA(50)
            if (close[i] > camarilla_r3[i] and volume_spike[i] and close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below Camarilla S3 + volume spike + price < 4h EMA(50)
            elif (close[i] < camarilla_s3[i] and volume_spike[i] and close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla S3 OR price below 4h EMA(50)
            if (close[i] < camarilla_s3[i] or close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R3 OR price above 4h EMA(50)
            if (close[i] > camarilla_r3[i] or close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals