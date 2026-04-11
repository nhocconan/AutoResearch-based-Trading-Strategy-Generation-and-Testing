#!/usr/bin/env python3
# 1h_4h_1d_rsi_trend_v1
# Strategy: 1h RSI mean reversion with 4h trend and 1d volume filter
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: In strong trends (4h EMA50), RSI extremes on 1h offer high-probability mean reversion entries. 1d volume filter ensures participation. Works in bull/bear by trading pullbacks in trend direction. Targets 15-30 trades/year via tight entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_rsi_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (precomputed)
    hours = prices.index.hour
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d volume average (20-period) for filter
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # 1h RSI(14) for entry signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_avg_20_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Trend filter
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        # Volume filter: current 1d volume > 20-day average (use aligned values)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)
        vol_filter = vol_1d_aligned[i] > vol_avg_20_1d_aligned[i] if not np.isnan(vol_1d_aligned[i]) else False
        
        # Entry conditions: RSI extremes in trend direction with volume
        # Long: RSI < 30 (oversold) in uptrend + volume
        # Short: RSI > 70 (overbought) in downtrend + volume
        if uptrend and rsi[i] < 30 and vol_filter and position != 1:
            position = 1
            signals[i] = 0.20
        elif downtrend and rsi[i] > 70 and vol_filter and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit: RSI returns to neutral zone (40-60)
        elif position == 1 and rsi[i] > 40:
            position = 0
            signals[i] = 0.0
        elif position == -1 and rsi[i] < 60:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals