#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation
# RSI < 30 for long, RSI > 70 for short in ranging markets.
# 4h EMA(50) filter ensures trades align with higher timeframe trend.
# Volume > 1.3x 20-period average confirms momentum.
# Session filter (08-20 UTC) reduces noise.
# Target: 15-35 trades/year per symbol to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Load 4h data for trend filter (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready or outside session
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_avg_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold + 4h uptrend + volume confirmation
            if (rsi[i] < 30 and 
                close[i] > ema_50_4h_aligned[i] and 
                volume[i] > 1.3 * vol_avg_20[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought + 4h downtrend + volume confirmation
            elif (rsi[i] > 70 and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume[i] > 1.3 * vol_avg_20[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI returns to neutral range or trend reversal
            if position == 1:
                # Exit long: RSI > 50 or trend turns down
                if (rsi[i] > 50 or close[i] < ema_50_4h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                # Exit short: RSI < 50 or trend turns up
                if (rsi[i] < 50 or close[i] > ema_50_4h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_RSI14_4hEMA50_VolumeConfirm_Session"
timeframe = "1h"
leverage = 1.0