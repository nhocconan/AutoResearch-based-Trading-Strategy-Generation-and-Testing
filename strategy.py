#!/usr/bin/env python3
"""
1h RSI(2) Mean Reversion with 4h Trend Filter and Session Filter
Strategy buys when RSI(2) < 10 in uptrend (price > 4h EMA50) and sells when RSI(2) > 90 in downtrend (price < 4h EMA50).
Uses 4h EMA50 for trend filter and restricts trading to 08-20 UTC to avoid low-liquidity hours.
Target: 15-30 trades/year per symbol with disciplined entries to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate RSI(2) on 1h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100).values  # fill NaN with 100 for first bar
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for EMA50 and RSI(2)
    
    for i in range(start_idx, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_50_4h_aligned[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: RSI(2) < 10 (oversold) in uptrend (price > 4h EMA50)
            if rsi_val < 10 and price > ema_trend:
                signals[i] = 0.20
                position = 1
            # Short: RSI(2) > 90 (overbought) in downtrend (price < 4h EMA50)
            elif rsi_val > 90 and price < ema_trend:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI(2) > 50 (mean reversion complete) or trend change
            if rsi_val > 50 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI(2) < 50 (mean reversion complete) or trend change
            if rsi_val < 50 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI2_MeanReversion_4hEMA50Trend_Session"
timeframe = "1h"
leverage = 1.0