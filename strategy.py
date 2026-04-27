#3/17/2025
#!/usr/bin/env python3
"""
1h_HybridTrendMean_4hTrendFilter
Hypothesis: Combines 4h trend direction (via EMA20/50 cross) with 1h mean reversion 
(RSI pullback to EMA20) for high-probability entries. Uses volume confirmation 
and session filter (08-20 UTC) to reduce noise. Designed for 15-25 trades/year 
to minimize fee drag while capturing trend continuations in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA20 and EMA50 for trend direction
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Trend: 1 = uptrend (EMA20 > EMA50), -1 = downtrend (EMA20 < EMA50)
    trend_4h = np.where(ema20_4h > ema50_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # 1h EMA20 for mean reversion entry
    ema20_1h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1h RSI(14) for overbought/oversold conditions
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need enough data for EMA20, RSI, and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trend_4h_aligned[i]) or np.isnan(ema20_1h[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        trend = trend_4h_aligned[i]
        ema20_val = ema20_1h[i]
        rsi_val = rsi[i]
        vol_conf = vol_confirm[i]
        in_session = session_filter[i]
        
        if position == 0:
            # Long: uptrend + RSI oversold pullback to EMA20 with volume
            if (trend == 1 and rsi_val < 30 and close[i] > ema20_val and 
                vol_conf and in_session):
                signals[i] = size
                position = 1
            # Short: downtrend + RSI overbought pullback to EMA20 with volume
            elif (trend == -1 and rsi_val > 70 and close[i] < ema20_val and 
                  vol_conf and in_session):
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: RSI overbought or trend change
            if rsi_val > 70 or trend == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI oversold or trend change
            if rsi_val < 30 or trend == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_HybridTrendMean_4hTrendFilter"
timeframe = "1h"
leverage = 1.0