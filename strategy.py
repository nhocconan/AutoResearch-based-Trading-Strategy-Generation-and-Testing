#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(2) mean reversion with 4h trend filter and session filter.
Long when 1h RSI(2) < 10 AND price > 4h EMA(50) AND session 08-20 UTC.
Short when 1h RSI(2) > 90 AND price < 4h EMA(50) AND session 08-20 UTC.
Exit when RSI(2) reverts to 50 OR session ends.
Uses 4h EMA for trend filter to avoid counter-trend trades, RSI(2) for extreme mean reversion,
and session filter to reduce noise. Target: 60-150 total trades over 4 years (15-37/year).
Works in bull markets (buys dips in uptrend) and bear markets (sells rallies in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA(50) on 4h timeframe
    close_4h_series = pd.Series(close_4h)
    ema_4h = close_4h_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate RSI(2) on 1h timeframe
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/2, min_periods=2, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/2, min_periods=2, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        in_session = (8 <= hours[i] <= 20)
        price = close[i]
        ema_trend = ema_4h_aligned[i]
        rsi_val = rsi[i]
        
        if position == 0 and in_session:
            # Long: RSI(2) < 10 (extreme oversold) AND price > 4h EMA(50) (uptrend)
            if rsi_val < 10 and price > ema_trend:
                signals[i] = 0.20
                position = 1
            # Short: RSI(2) > 90 (extreme overbought) AND price < 4h EMA(50) (downtrend)
            elif rsi_val > 90 and price < ema_trend:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI(2) > 50 (mean reversion) OR session end
            if rsi_val > 50 or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI(2) < 50 (mean reversion) OR session end
            if rsi_val < 50 or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI2_MeanReversion_4hEMA50_SessionFilter"
timeframe = "1h"
leverage = 1.0