#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and session timing.
- Primary timeframe: 1h for entries/exits.
- HTF: 4h EMA(50) for trend direction (bullish if price > EMA50, bearish if price < EMA50).
- RSI: 14-period RSI on 1h for mean reversion signals.
- Session filter: Only trade during 08-20 UTC to avoid low-liquidity hours.
- Entry: Long when RSI < 30 AND 4h EMA50 trend bullish AND in session.
         Short when RSI > 70 AND 4h EMA50 trend bearish AND in session.
- Exit: Opposite RSI signal (RSI > 50 for long exit, RSI < 50 for short exit) or loss of trend.
- Signal size: 0.20 discrete to limit drawdown and reduce fee churn.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    
    # Calculate RSI(14) on 1h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Fill NaN with neutral 50
    
    # Get 4h data for EMA(50) trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h close
    ema_50 = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1h
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need enough 4h bars for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_50_val = ema_50_aligned[i]
        curr_close = close[i]
        curr_rsi = rsi[i]
        session_ok = in_session[i]
        
        if position == 0:
            # Check for entry signals
            if session_ok:
                # Bullish mean reversion: RSI oversold AND 4h EMA50 bullish (price > EMA50)
                if curr_rsi < 30 and curr_close > ema_50_val:
                    signals[i] = 0.20
                    position = 1
                # Bearish mean reversion: RSI overbought AND 4h EMA50 bearish (price < EMA50)
                elif curr_rsi > 70 and curr_close < ema_50_val:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: RSI bullish (>50) OR loss of 4h bullish trend
            if curr_rsi > 50 or curr_close < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI bearish (<50) OR loss of 4h bearish trend
            if curr_rsi < 50 or curr_close > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI14_4hEMA50Trend_MeanReversion_Session_v1"
timeframe = "1h"
leverage = 1.0