#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(2) Extreme Reversion + 4h/1d Trend Filter + Session Filter.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend) and 1d EMA200 for higher-timeframe bias.
- Entry: Long when RSI(2) < 10 AND price > 4h EMA50 AND price > 1d EMA200 AND session 08-20 UTC;
         Short when RSI(2) > 90 AND price < 4h EMA50 AND price < 1d EMA200 AND session 08-20 UTC.
- Exit: Long exits when RSI(2) > 60; Short exits when RSI(2) < 40.
- Signal size: 0.20 discrete to minimize fee churn.
- RSI(2) captures short-term overextremes; 4h/1d EMAs ensure trading with higher-timeframe trend; session filter reduces noise.
- Works in bull (buying dips in uptrend) and bear (selling rallies in downtrend) with controlled trade frequency.
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
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for EMA200 higher-timeframe filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate RSI(2) on 1h close
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Pre-compute session filter (08-20 UTC)
    # open_time is already datetime64[ms], use DatetimeIndex.hour
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 200)  # 4h EMA50 needs 50, 1d EMA200 needs 200
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_rsi = rsi[i]
        session_ok = in_session[i]
        
        if position == 0:
            # Check for entry signals
            if curr_rsi < 10 and curr_close > ema_50_4h_aligned[i] and curr_close > ema_200_1d_aligned[i] and session_ok:
                # Long: RSI oversold in uptrend
                signals[i] = 0.20
                position = 1
            elif curr_rsi > 90 and curr_close < ema_50_4h_aligned[i] and curr_close < ema_200_1d_aligned[i] and session_ok:
                # Short: RSI overbought in downtrend
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: exit when RSI > 60 (mean reversion complete)
            if curr_rsi > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit when RSI < 40 (mean reversion complete)
            if curr_rsi < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI2_Extreme_4hEMA50_1dEMA200_Session_v1"
timeframe = "1h"
leverage = 1.0