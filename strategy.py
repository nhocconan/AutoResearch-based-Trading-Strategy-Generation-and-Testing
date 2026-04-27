#!/usr/bin/env python3
"""
1h_RSI4060_MeanReversion_4hTrendFilter
Hypothesis: Mean reversion on 1h with RSI 40-60 bands, filtered by 4h trend (EMA50) and volume confirmation.
Trades only during London/NY session (08-20 UTC) to avoid Asian session noise.
Targets 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
Uses RSI mean reversion in ranging markets and trend alignment for directional bias.
Works in bull via long bias in uptrend, bear via short bias in downtrend, range via mean reversion.
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
    
    # Calculate RSI(14) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Session filter: 08-20 UTC (London + NY overlap)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need enough data for RSI and EMA
    start_idx = max(14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_confirm[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        ema_50_val = ema_50_4h_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: RSI < 40 (oversold) in uptrend (price > EMA50) with volume
            if rsi_val < 40 and close[i] > ema_50_val and vol_conf:
                signals[i] = size
                position = 1
            # Short: RSI > 60 (overbought) in downtrend (price < EMA50) with volume
            elif rsi_val > 60 and close[i] < ema_50_val and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion complete) or trend change
            if rsi_val > 50 or close[i] < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI < 50 (mean reversion complete) or trend change
            if rsi_val < 50 or close[i] > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_RSI4060_MeanReversion_4hTrendFilter"
timeframe = "1h"
leverage = 1.0