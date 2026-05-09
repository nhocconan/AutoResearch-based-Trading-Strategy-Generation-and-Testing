#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h EMA50 trend filter and volume confirmation
# Uses 4h EMA50 for trend direction and 1h RSI for entry timing to capture pullbacks
# in both bull and bear markets. Volume filter reduces false signals. Session filter (08-20 UTC)
# avoids low-liquidity periods. Target: 15-37 trades/year (60-150 over 4 years).
name = "1h_RSI14_4hEMA50_Trend_Volume_Session"
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
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1h = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h RSI(14)
    delta = pd.Series(close).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for RSI and EMA calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1h[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        trend_up = close[i] > ema_50_1h[i]
        trend_down = close[i] < ema_50_1h[i]
        
        if position == 0:
            # Long: RSI oversold + uptrend + volume confirmation
            if rsi_oversold and trend_up and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought + downtrend + volume confirmation
            elif rsi_overbought and trend_down and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought or trend reversal
            if rsi[i] > 70 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI oversold or trend reversal
            if rsi[i] < 30 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals