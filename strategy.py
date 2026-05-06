#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d trend alignment with volume confirmation
# Long when 4h EMA21 > 4h EMA50, 1d close > 1d EMA50, price pulls back to 4h EMA21 with volume spike
# Short when 4h EMA21 < 4h EMA50, 1d close < 1d EMA50, price bounces to 4h EMA21 with volume spike
# Uses higher timeframe for trend direction (4h/1d) and 1h for precise entry on pullbacks
# Volume spike confirms institutional interest during retracements
# Target: 15-37 trades/year (60-150 over 4 years) with 0.20 position sizing

name = "1h_4h1dEMA21_50_PullbackVolume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA21 and EMA50 ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMAs to 1h timeframe
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1d EMA50 ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 1h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: >2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema21_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h uptrend (EMA21 > EMA50), 1d uptrend (close > EMA50), pullback to 4h EMA21 with volume spike
            if (ema21_4h_aligned[i] > ema50_4h_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and
                low[i] <= ema21_4h_aligned[i] * 1.005 and  # Allow small tolerance for pullback
                high[i] >= ema21_4h_aligned[i] * 0.995 and
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend (EMA21 < EMA50), 1d downtrend (close < EMA50), bounce to 4h EMA21 with volume spike
            elif (ema21_4h_aligned[i] < ema50_4h_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and
                  high[i] >= ema21_4h_aligned[i] * 0.995 and  # Allow small tolerance for bounce
                  low[i] <= ema21_4h_aligned[i] * 1.005 and
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: 4h trend breaks down (EMA21 < EMA50) or 1d trend breaks down
            if ema21_4h_aligned[i] < ema50_4h_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: 4h trend breaks up (EMA21 > EMA50) or 1d trend breaks up
            if ema21_4h_aligned[i] > ema50_4h_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals