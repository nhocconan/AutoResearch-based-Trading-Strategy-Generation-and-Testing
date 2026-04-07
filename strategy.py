#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour RSI pullback with 4-hour EMA trend filter and volume confirmation
# Uses 4-hour EMA for trend direction (avoids whipsaw in 1h), 1-hour RSI for entry timing
# Volume filter ensures breakouts have conviction. Designed for low trade frequency (15-30/year)
# Works in bull/bear via trend filter + oversold/overbought entries during pullbacks

name = "1h_rsi_pullback_4h_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4-hour data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1-hour RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Average volume for volume confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 8-20 UTC (avoid low-volume Asian session)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Only trade during active session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: RSI > 70 (overbought) or trend turns down
            if rsi[i] > 70 or ema_50_4h_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI < 30 (oversold) or trend turns up
            if rsi[i] < 30 or ema_50_4h_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Trend filter: 4h EMA(50) slope
            uptrend = ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]
            downtrend = ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1]
            
            # Volume confirmation
            volume_confirm = volume[i] > 1.5 * vol_avg[i]
            
            # Long: RSI < 30 (oversold) in uptrend with volume
            if rsi[i] < 30 and uptrend and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought) in downtrend with volume
            elif rsi[i] > 70 and downtrend and volume_confirm:
                signals[i] = -0.20
                position = -1
    
    return signals