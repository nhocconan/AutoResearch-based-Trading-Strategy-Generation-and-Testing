#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and session filter (08-20 UTC).
Long when 4h close > 4h EMA50 (bullish trend) AND RSI(14) < 30 (oversold) during active session.
Short when 4h close < 4h EMA50 (bearish trend) AND RSI(14) > 70 (overbought) during active session.
Exit when RSI returns to 50 (mean reversion complete).
Uses 4h for trend direction to avoid counter-trend whipsaw, 1h RSI for precise entry timing.
Session filter reduces noise trades during low-liquidity hours.
Targets 60-120 total trades over 4 years (15-30/year) to minimize fee drag.
Works in bull markets (buys dips in uptrend) and bear markets (sells rallies in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA50 on 4h timeframe for trend filter
    close_4h_series = pd.Series(close_4h)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate RSI(14) on 1h timeframe for mean reversion signals
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    # Precompute session hours (08-20 UTC)
    hours = prices.index.hour
    
    for i in range(start_idx, n):
        # Skip if required data is not available
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi_values[i]):
            signals[i] = 0.0
            continue
        
        ema_50 = ema_50_4h_aligned[i]
        rsi_val = rsi_values[i]
        price = close[i]
        hour = hours[i]
        
        # Session filter: only trade during 08-20 UTC
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish 4h trend AND RSI oversold (<30)
            if price > ema_50 and rsi_val < 30:
                signals[i] = 0.20
                position = 1
            # Short: bearish 4h trend AND RSI overbought (>70)
            elif price < ema_50 and rsi_val > 70:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (50)
            if rsi_val >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI returns to neutral (50)
            if rsi_val <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI14_MeanRev_4hEMA50Trend_Session"
timeframe = "1h"
leverage = 1.0