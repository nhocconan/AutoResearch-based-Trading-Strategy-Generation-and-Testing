#!/usr/bin/env python3
# 6h_1d_pullback_reversal_v1
# Hypothesis: On 6b, enter pullbacks in the direction of the 1d trend using RSI(2) for entry timing and volume confirmation.
# Works in bull/bear by aligning with higher timeframe trend. Uses RSI(2) to catch deep pullbacks in strong trends.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_1d_pullback_reversal_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d trend direction: 1 if close > EMA50, -1 if close < EMA50
    trend_1d = np.where(close_1d > ema50_1d, 1, -1)
    
    # Align 1d trend to 6h
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # RSI(2) for entry timing
    def calculate_rsi(close, period=2):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period-1] = np.mean(gain[1:period]) if period < len(gain) else np.nan
        avg_loss[period-1] = np.mean(loss[1:period]) if period < len(loss) else np.nan
        
        for i in range(period, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi2 = calculate_rsi(close)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(rsi2[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in session and with volume confirmation
        if not (in_session[i] and vol_confirm[i]):
            if position != 0:
                # Hold position until exit signal
                pass
            else:
                signals[i] = 0.0
                continue
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or trend changes
            if rsi2[i] > 70 or trend_1d_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or trend changes
            if rsi2[i] < 30 or trend_1d_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Enter pullback in direction of 1d trend
            if trend_1d_aligned[i] == 1:  # 1d uptrend
                # Long on RSI(2) < 10 (deep pullback)
                if rsi2[i] < 10:
                    position = 1
                    signals[i] = 0.25
            elif trend_1d_aligned[i] == -1:  # 1d downtrend
                # Short on RSI(2) > 90 (strong pullback)
                if rsi2[i] > 90:
                    position = -1
                    signals[i] = -0.25
    
    return signals