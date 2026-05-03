#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Long when price breaks above Camarilla R3, price > 4h EMA50, and volume > 2.0x 20-bar average
# Short when price breaks below Camarilla S3, price < 4h EMA50, and volume > 2.0x 20-bar average
# Uses 4h EMA for higher timeframe trend alignment (matches experiment HTF)
# Volume spike confirms breakout strength
# Discrete position sizing (0.20) to minimize fee churn
# Designed for low trade frequency (15-37/year on 1h) to avoid fee drag
# Session filter: 08-20 UTC to reduce noise trades
# Works in bull (breakouts above rising EMA) and bear (breakdowns below falling EMA)

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_Volume_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA(50) trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA to 1h timeframe (wait for completed 4h bar)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels on 4h from previous 4h OHLC
    # Camarilla: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    # We use previous period's OHLC by shifting
    df_4h_shifted = df_4h.copy()
    df_4h_shifted['open'] = df_4h_shifted['open'].shift(1)
    df_4h_shifted['high'] = df_4h_shifted['high'].shift(1)
    df_4h_shifted['low'] = df_4h_shifted['low'].shift(1)
    df_4h_shifted['close'] = df_4h_shifted['close'].shift(1)
    
    # Calculate Camarilla R3 and S3 from previous 4h
    camarilla_r3_4h = df_4h_shifted['close'] + 1.1 * (df_4h_shifted['high'] - df_4h_shifted['low'])
    camarilla_s3_4h = df_4h_shifted['close'] - 1.1 * (df_4h_shifted['high'] - df_4h_shifted['low'])
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h_shifted, camarilla_r3_4h.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h_shifted, camarilla_s3_4h.values)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Session filter: 08-20 UTC (pre-compute hours array)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(50, 20) + 1  # EMA(50) + Donchian(20) + volume MA(20) warmup + shift(1)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Camarilla R3, price > 4h EMA50, volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price < Camarilla S3, price < 4h EMA50, volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Camarilla S3 or price < 4h EMA50
            if (close[i] < camarilla_s3_aligned[i] or 
                close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price > Camarilla R3 or price > 4h EMA50
            if (close[i] > camarilla_r3_aligned[i] or 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals