#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and session filter.
# Long when 1h RSI < 30 AND 4h EMA50 > EMA200 (uptrend) AND hour in 08-20 UTC.
# Short when 1h RSI > 70 AND 4h EMA50 < EMA200 (downtrend) AND hour in 08-20 UTC.
# Exit when 1h RSI returns to 50 (mean reversion completion).
# Uses discrete position size 0.20. Session filter reduces noise trades, 4h EMA alignment ensures trend direction.
# Target: 60-150 total trades over 4 years (15-37/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC) - avoids datetime64 arithmetic issues
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data once before loop for EMA calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # === 4h Indicators: EMA50 and EMA200 for trend filter ===
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h EMAs to 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_4h, ema_200)
    
    # === 1h Indicators: RSI(14) for mean reversion signal ===
    # Calculate RSI using Wilder's smoothing (standard RSI)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing: alpha = 1/period
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(np.isnan(rsi), 50, rsi)  # Handle division by zero
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        ema_50_val = ema_50_aligned[i]
        ema_200_val = ema_200_aligned[i]
        rsi_val = rsi[i]
        price = close[i]
        
        # Trend filter: 4h EMA50 > EMA200 for uptrend, < for downtrend
        uptrend = ema_50_val > ema_200_val
        downtrend = ema_50_val < ema_200_val
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if RSI returns to 50 (mean reversion completion)
            if rsi_val >= 50:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if RSI returns to 50 (mean reversion completion)
            if rsi_val <= 50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: RSI < 30 (oversold) in uptrend session
            if rsi_val < 30 and uptrend:
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: RSI > 70 (overbought) in downtrend session
            elif rsi_val > 70 and downtrend:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_RSI_MeanReversion_4hEMATrend_SessionFilter_V1"
timeframe = "1h"
leverage = 1.0