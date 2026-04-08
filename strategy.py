#!/usr/bin/env python3
# 1h_rsi_ema_4h_filter_v1
# Hypothesis: RSI(14) pullback to EMA(21) on 1h with 4h EMA(50) trend filter and volume confirmation captures mean-reversion entries during pullbacks in both bull and bear markets. Uses 4h trend for direction, 1h for entry timing, and session filter (08-20 UTC) to reduce noise. Target: 15-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_ema_4h_filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # 4h EMA50 for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h EMA21
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume filter: 1h volume > 1.2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available or outside session
        if (np.isnan(ema_21[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 60 or price < EMA21
            if (rsi[i] > 60) or (close[i] < ema_21[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI < 40 or price > EMA21
            if (rsi[i] < 40) or (close[i] > ema_21[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: RSI < 40, price > EMA21, 4h uptrend, volume
            if (rsi[i] < 40) and (close[i] > ema_21[i]) and (close[i] > ema_50_4h_aligned[i]) and volume_filter[i]:
                position = 1
                signals[i] = 0.20
            # Short entry: RSI > 60, price < EMA21, 4h downtrend, volume
            elif (rsi[i] > 60) and (close[i] < ema_21[i]) and (close[i] < ema_50_4h_aligned[i]) and volume_filter[i]:
                position = -1
                signals[i] = -0.20
    
    return signals