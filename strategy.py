#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(2) mean reversion with 4h trend filter and session timing
# RSI(2) captures extreme short-term momentum exhaustion for mean reversion entries.
# 4h EMA(50) provides trend bias: long only when price > EMA50, short only when price < EMA50.
# Session filter (08-20 UTC) avoids low-liquidity Asian session noise.
# Target 15-37 trades/year by requiring confluence of RSI extreme + trend alignment + session.
# Works in bull/bear markets: in uptrends, buy RSI(2) pullbacks; in downtrends, sell RSI(2) bounces.

name = "1h_RSI2_4hEMA50_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend bias
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h RSI(2) for mean reversion signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_2 = 100 - (100 / (1 + rs))
    rsi_2_values = rsi_2.fillna(50).values  # fill NaN with 50 (neutral)
    
    # Session filter: 08-20 UTC (avoid low-liquidity sessions)
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Start after RSI(2) and EMA(50) warmup
    start_idx = max(2, 50)
    
    for i in range(start_idx, n):
        # Skip if required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi_2_values[i])):
            signals[i] = 0.0
            continue
        
        # Only trade during session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_50_4h_aligned[i]
        rsi_val = rsi_2_values[i]
        
        # Mean reversion entries with trend filter
        if rsi_val < 10 and price > ema_trend:  # Oversold in uptrend -> long
            signals[i] = 0.20
        elif rsi_val > 90 and price < ema_trend:  # Overbought in downtrend -> short
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals