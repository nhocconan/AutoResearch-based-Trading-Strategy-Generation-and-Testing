#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA(50) trend filter and volume confirmation
# Long when price breaks above Camarilla R3 + volume spike + price > 12h EMA(50)
# Short when price breaks below Camarilla S3 + volume spike + price < 12h EMA(50)
# Uses Camarilla levels from previous 12h bar to avoid look-ahead
# 12h EMA(50) filter reduces whipsaw and captures medium-term trend
# Volume spike (2.0x 20-period average) confirms institutional participation
# Designed for low trade frequency (12-37/year on 6h) to minimize fee drag
# Works in both bull (breakouts) and bear (mean reversion at extremes) markets

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_Volume_v1"
timeframe = "6h"
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
    
    # Get 12h data for Camarilla levels and EMA(50) trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 6h timeframe (wait for completed 12h bar)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # Camarilla levels: based on previous bar's high, low, close
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(1, len(df_12h)):
        # Get the completed 12h bar values (index i-1)
        phigh = df_12h['high'].iloc[i-1]
        plow = df_12h['low'].iloc[i-1]
        pclose = df_12h['close'].iloc[i-1]
        
        # Calculate Camarilla levels for this completed bar
        r3 = pclose + (phigh - plow) * 1.1 / 4
        s3 = pclose - (phigh - plow) * 1.1 / 4
        
        # Find the corresponding 6h bar indices for this 12h bar
        # Each 12h bar spans 2 6h bars
        start_6h_idx = i * 2
        end_6h_idx = min((i + 1) * 2, n)
        
        if start_6h_idx < n:
            camarilla_r3[start_6h_idx:end_6h_idx] = r3
            camarilla_s3[start_6h_idx:end_6h_idx] = s3
    
    # Volume confirmation (2.0x 20-period average) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(50, 20)  # max(50 for 12h EMA, 20 for volume MA)
    
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