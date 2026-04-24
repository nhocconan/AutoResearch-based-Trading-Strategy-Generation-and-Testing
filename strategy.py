#!/usr/bin/env python3
"""
Hypothesis: 1h RSI mean reversion with 4h EMA50 trend filter and session filter (08-20 UTC).
- Primary timeframe: 1h targeting 60-120 total trades over 4 years (15-30/year).
- HTF: 4h for EMA50 trend filter.
- Entry: Long when RSI(14) < 30 AND price > 4h EMA50 AND session 08-20 UTC.
         Short when RSI(14) > 70 AND price < 4h EMA50 AND session 08-20 UTC.
- Exit: RSI crosses back to neutral (40 for long exit, 60 for short exit) OR session end.
- Signal size: 0.20 discrete to minimize fee drag.
- RSI mean reversion works in ranging markets; EMA50 filter ensures we trade with the higher timeframe trend.
- Session filter reduces noise during low-volume hours (20-08 UTC).
- Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
- Estimated trades: ~90 total over 4 years (~22/year) based on RSI extreme frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def rsi(close, period=14):
    """Calculate Relative Strength Index."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    
    # Calculate 4h trend filter: EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    ema50_4h = ema(df_4h['close'].values, 50)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h, additional_delay_bars=1)
    
    # Calculate RSI(14) on 1h
    rsi_values = rsi(close, 14)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where RSI is ready
    start_idx = 14
    
    for i in range(start_idx, n):
        # Skip if 4h EMA data not ready
        if np.isnan(ema50_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_rsi = rsi_values[i]
        session_active = in_session[i]
        
        # Exit conditions
        if position != 0:
            # Exit long: RSI crosses above 40 OR session ends
            if position == 1:
                if curr_rsi > 40 or not session_active:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: RSI crosses below 60 OR session ends
            elif position == -1:
                if curr_rsi < 60 or not session_active:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: RSI extreme with trend filter and session
        if position == 0 and session_active:
            # Long: RSI < 30 (oversold) AND bullish 4h trend (price > EMA50)
            if curr_rsi < 30.0 and curr_close > ema50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought) AND bearish 4h trend (price < EMA50)
            elif curr_rsi > 70.0 and curr_close < ema50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_RSI_MeanReversion_4hEMA50_TrendFilter_Session_v1"
timeframe = "1h"
leverage = 1.0