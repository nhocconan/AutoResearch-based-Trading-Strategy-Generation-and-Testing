#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and volume confirmation
# RSI < 30 for long, RSI > 70 for short in ranging markets
# 4h EMA20 filter: only take longs above EMA20, shorts below EMA20
# Volume spike (>1.5x 20-period average) confirms momentum
# Session filter: 08-20 UTC to avoid low-volume periods
# Target: 15-30 trades/year per symbol with disciplined entries
name = "1h_RSI_MeanReversion_4hEMA_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA20 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(volume_ma[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) + above 4h EMA20 + volume spike
            if (rsi[i] < 30 and 
                close[i] > ema_20_4h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought) + below 4h EMA20 + volume spike
            elif (rsi[i] > 70 and 
                  close[i] < ema_20_4h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if RSI > 50 (mean reversion complete) or price breaks below 4h EMA20
            if (rsi[i] > 50) or (close[i] < ema_20_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if RSI < 50 (mean reversion complete) or price breaks above 4h EMA20
            if (rsi[i] < 50) or (close[i] > ema_20_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals