#!/usr/bin/env python3
"""
Hypothesis: 1h RSI mean reversion with 4h trend filter and session filter.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h EMA50 for trend direction (bullish when close > EMA50, bearish when close < EMA50).
- Entry: Long when RSI(14) < 30 in 4h bull trend during 08-20 UTC session; Short when RSI(14) > 70 in 4h bear trend during 08-20 UTC session.
- Exit: RSI returns to neutral zone (40-60) or opposite RSI extreme.
- Signal size: 0.20 discrete to minimize fee churn.
- Designed for BTC/ETH: Mean reversion works in ranging markets, 4h trend filter avoids counter-trend trades, session filter reduces noise outside active hours.
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
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend
    ema_50 = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate RSI(14) on 1h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Pre-compute session filter (08-20 UTC)
    # open_time is already datetime64[ms], convert to get hour
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where RSI is ready
    start_idx = 14
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema_50_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_rsi = rsi_values[i]
        curr_close = close[i]
        
        # Determine 4h EMA50 trend
        trend_bullish = close[i] > ema_50_aligned[i]
        trend_bearish = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Check for entry signals only during session
            if in_session[i]:
                # Long: RSI < 30 (oversold) in 4h bull trend
                if curr_rsi < 30 and trend_bullish:
                    signals[i] = 0.20
                    position = 1
                # Short: RSI > 70 (overbought) in 4h bear trend
                elif curr_rsi > 70 and trend_bearish:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long position: exit when RSI returns to neutral (40-60) or breaks down
            if curr_rsi > 40 or curr_rsi < 30:  # Exit on recovery or further oversold
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit when RSI returns to neutral (40-60) or breaks up
            if curr_rsi < 60 or curr_rsi > 70:  # Exit on recovery or further overbought
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_MeanReversion_4hEMA50_Trend_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0