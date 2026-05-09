#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1w timeframe for long-term direction and structure.
# Uses 1w Bollinger Bands (20, 2) for breakouts with 1w EMA20 trend filter and volume confirmation.
# Designed to capture major trends while filtering counter-trend noise in both bull and bear markets.
# Target: 20-35 trades per year to minimize fee drag and improve generalization.
name = "4h_BollingerBreakout_1wEMA20_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Bollinger Bands and EMA20
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA20 trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 1w Bollinger Bands (20, 2)
    close_1w = df_1w['close'].values
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_1w = sma_20_1w + 2.0 * std_20_1w
    lower_1w = sma_20_1w - 2.0 * std_20_1w
    
    # Align 1w Bollinger Bands and EMA20 to 4h timeframe
    upper_4h = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_4h = align_htf_to_ltf(prices, df_1w, lower_1w)
    
    # Volume filter: spike above 2.0x 28-period average (1.16 days of 4h bars)
    vol_ma = pd.Series(volume).rolling(window=28, min_periods=28).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 28)  # Wait for EMA20 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_4h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(upper_4h[i]) or np.isnan(lower_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation
        
        # Pre-compute hour for session filter (UTC 0-24)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Trade during active hours (8 AM - 8 PM UTC) to avoid low liquidity
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: price above 1w Upper BB, 1w uptrend (price > EMA20), volume breakout
            if (close[i] > upper_4h[i] and 
                close[i] > ema_20_4h[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: price below 1w Lower BB, 1w downtrend (price < EMA20), volume breakdown
            elif (close[i] < lower_4h[i] and 
                  close[i] < ema_20_4h[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below 1w Lower BB or trend reversal
            if close[i] < lower_4h[i] or close[i] < ema_20_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above 1w Upper BB or trend reversal
            if close[i] > upper_4h[i] or close[i] > ema_20_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals