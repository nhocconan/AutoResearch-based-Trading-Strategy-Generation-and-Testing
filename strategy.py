#!/usr/bin/env python3
# 1h_mtf_rsi_reversal_v1
# Hypothesis: Mean reversion on 1h timeframe using RSI with 4h/1d trend filter.
# Long when 1h RSI < 30 and 4h trend is up (price > 4h EMA50) and 1d trend is up (price > 1d EMA50).
# Short when 1h RSI > 70 and 4h trend is down (price < 4h EMA50) and 1d trend is down (price < 1d EMA50).
# Exit when 1h RSI returns to neutral (40-60 range) or trend changes.
# Uses 4h/1d for trend direction, 1h for entry timing to avoid overtrading.
# Session filter: 08-20 UTC to reduce noise.
# Position size: 0.20 (20% of capital) to manage drawdown.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_mtf_rsi_reversal_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-calculate hour filter for session (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1h RSI (14 period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Need enough data for EMA50 and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                pass  # Hold position outside session
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral (40-60) or trend turns down
            if (rsi[i] >= 40 and rsi[i] <= 60) or (close[i] < ema_4h_aligned[i]) or (close[i] < ema_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral (40-60) or trend turns up
            if (rsi[i] >= 40 and rsi[i] <= 60) or (close[i] > ema_4h_aligned[i]) or (close[i] > ema_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: RSI oversold (<30) and both trends up
            if (rsi[i] < 30 and 
                close[i] > ema_4h_aligned[i] and 
                close[i] > ema_1d_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: RSI overbought (>70) and both trends down
            elif (rsi[i] > 70 and 
                  close[i] < ema_4h_aligned[i] and 
                  close[i] < ema_1d_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals