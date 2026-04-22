#!/usr/bin/env python3

"""
Hypothesis: 1-hour RSI mean reversion with 4h trend filter and volume confirmation.
In strong trends (4h), price pulls back to RSI(40) in uptrends or RSI(60) in downtrends.
Volume spike confirms the bounce. This captures counter-trend moves within the trend,
working in both bull and bear markets by trading with the 4h trend.
Target: 15-37 trades/year per symbol (60-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate 1h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend
        uptrend = close[i] > ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: uptrend + RSI < 40 (pullback) + volume spike
            if (uptrend and 
                rsi[i] < 40 and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.20
                position = 1
            # Short: downtrend + RSI > 60 (pullback) + volume spike
            elif ((not uptrend) and 
                  rsi[i] > 60 and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI returns to neutral (50) or opposite extreme
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI >= 50
                if rsi[i] >= 50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI <= 50
                if rsi[i] <= 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_RSI_MeanReversion_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0