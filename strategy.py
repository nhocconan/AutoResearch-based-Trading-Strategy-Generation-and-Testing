#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(2) mean reversion with 4h trend filter and session timing.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA50 trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- RSI(2): Oversold <10 for longs, overbought >90 for shorts in 1h timeframe.
- Session filter: Only trade 08-20 UTC to avoid low-volume Asian session noise.
- Entry: Long when RSI(2) < 10 AND price > 4h EMA50 AND session open.
         Short when RSI(2) > 90 AND price < 4h EMA50 AND session open.
- Exit: RSI(2) > 60 for long exit, RSI(2) < 40 for short exit (mean reversion completion).
- Signal size: 0.20 discrete to minimize fee drag.
- Works in bull markets via longs in uptrends, bear markets via shorts in downtrends.
- Avoids choppy markets via strong 4h trend filter (only trade with 4h momentum).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h RSI(2) for mean reversion signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = pd.Series(gain).ewm(span=2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(span=2, adjust=False, min_periods=2).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where RSI is ready
    start_idx = 2  # RSI(2) needs 2 periods
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if EMA50 data not ready
        if np.isnan(ema_50_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_rsi = rsi[i]
        
        # Exit conditions: RSI mean reversion completion
        if position != 0:
            # Exit long: RSI > 60 (overbought in short term)
            if position == 1:
                if curr_rsi > 60:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: RSI < 40 (oversold in short term)
            elif position == -1:
                if curr_rsi < 40:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: RSI extremes with 4h trend filter
        if position == 0:
            # Long: RSI < 10 (extremely oversold) AND price > 4h EMA50 (uptrend)
            long_condition = (curr_rsi < 10 and curr_close > ema_50_4h_aligned[i])
            
            # Short: RSI > 90 (extremely overbought) AND price < 4h EMA50 (downtrend)
            short_condition = (curr_rsi > 90 and curr_close < ema_50_4h_aligned[i])
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_RSI2_MeanReversion_4hEMA50Trend_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0