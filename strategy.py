#!/usr/bin/env python3
"""
1h_RSI_MeanReversion_4hTrendFilter
Hypothesis: Mean reversion on 1h RSI extremes (oversold/overbought) with 4h EMA trend filter to avoid counter-trend trades. Works in ranging markets (bear/consolidation) via reversals and in bull markets via pullbacks to trend.
Target: 15-30 trades/year (60-120 total over 4 years) using discrete sizing (0.20) and session filter (08-20 UTC) to reduce noise.
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
    volume = prices['volume'].values
    
    # 1h RSI(14) for mean reversion signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values
    
    # 4h EMA(50) for trend filter (avoid counter-trend)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08:00-20:00 UTC (reduce noise outside active hours)
    hours = prices.index.hour  # DatetimeIndex from parquet
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for RSI (14), EMA50 (50), volume MA (20)
    start_idx = max(14, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi_values[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Require session filter and volume spike for entry precision
        if not (in_session[i] and volume_spike[i]):
            # Hold current position if any, but no new entries outside filters
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) + price above 4h EMA (uptrend bias)
            long_setup = (rsi_values[i] < 30) and (close[i] > ema_50_4h_aligned[i])
            # Short: RSI overbought (>70) + price below 4h EMA (downtrend bias)
            short_setup = (rsi_values[i] > 70) and (close[i] < ema_50_4h_aligned[i])
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold until RSI reverts to neutral (50) or trend breaks
            signals[i] = 0.20
            if (rsi_values[i] >= 50) or (close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold until RSI reverts to neutral (50) or trend breaks
            signals[i] = -0.20
            if (rsi_values[i] <= 50) or (close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_RSI_MeanReversion_4hTrendFilter"
timeframe = "1h"
leverage = 1.0