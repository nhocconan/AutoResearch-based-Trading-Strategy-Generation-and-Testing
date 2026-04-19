#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour EMA crossover with 4-hour MACD trend filter and volume spike confirmation.
# Long when: 9 EMA crosses above 21 EMA, 4-hour MACD > 0, volume > 2x 20-period average, and time in 08-20 UTC session.
# Short when: 9 EMA crosses below 21 EMA, 4-hour MACD < 0, volume > 2x 20-period average, and time in 08-20 UTC session.
# Exit when: Opposite EMA crossover occurs.
# This uses 4h MACD for trend direction (reduces whipsaw) and 1h EMA for entry timing.
# Session filter reduces noise outside active hours. Volume spike confirms momentum.
# Target: 20-30 trades/year per symbol (60-120 over 4 years).
name = "1h_EMA9_21_Cross_4hMACD_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4-hour data for MACD trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate MACD on 4-hour data
    ema12_4h = pd.Series(close_4h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26_4h = pd.Series(close_4h).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_4h = ema12_4h - ema26_4h
    signal_4h = pd.Series(macd_4h).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist_4h = macd_4h - signal_4h  # Using MACD histogram for cleaner signals
    
    # Align 4H MACD to 1H timeframe
    macd_hist_4h_aligned = align_htf_to_ltf(prices, df_4h, macd_hist_4h)
    
    # 1-hour EMA for entry timing
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 26)  # Wait for EMA21 and EMA26 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema9[i]) or np.isnan(ema21[i]) or 
            np.isnan(macd_hist_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # EMA crossover signals
        ema9_prev = ema9[i-1]
        ema21_prev = ema21[i-1]
        ema9_curr = ema9[i]
        ema21_curr = ema21[i]
        
        bullish_cross = (ema9_prev <= ema21_prev) and (ema9_curr > ema21_curr)
        bearish_cross = (ema9_prev >= ema21_prev) and (ema9_curr < ema21_curr)
        
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        vol_spike = vol > 2.0 * vol_ma
        
        macd_hist = macd_hist_4h_aligned[i]
        
        if position == 0:
            # Long entry: bullish EMA crossover, 4h MACD positive, volume spike, in session
            if bullish_cross and macd_hist > 0 and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short entry: bearish EMA crossover, 4h MACD negative, volume spike, in session
            elif bearish_cross and macd_hist < 0 and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: bearish EMA crossover
            if bearish_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: bullish EMA crossover
            if bullish_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals