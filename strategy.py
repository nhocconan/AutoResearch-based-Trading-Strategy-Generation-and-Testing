#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h RSI pullback with 4h trend filter and 1d volume confirmation
# Hypothesis: In uptrends, buy RSI pullbacks to 40; in downtrends, sell RSI bounces to 60.
# 4h EMA50 defines trend, 1d volume filter ensures participation, RSI(14) provides entry timing.
# Works in bull via pullback longs, in bear via bounce shorts. Low turnover by requiring trend alignment.
name = "1h_rsi_pullback_4h_trend_1d_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for volume confirmation (20-period average)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate RSI(14) on 1h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available or outside session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(rsi[i]) or not session_mask[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1-day average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 60 (overbought) or trend breaks down
            if rsi[i] > 60 or close[i] < ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
        elif position == -1:  # Short position
            # Exit: RSI < 40 (oversold) or trend breaks up
            if rsi[i] < 40 or close[i] > ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: uptrend (price > EMA50) + RSI pullback to 40 + volume confirmation
            if close[i] > ema_4h_aligned[i] and rsi[i] < 40 and vol_confirm:
                position = 1
                signals[i] = 0.20
            # Enter short: downtrend (price < EMA50) + RSI bounce to 60 + volume confirmation
            elif close[i] < ema_4h_aligned[i] and rsi[i] > 60 and vol_confirm:
                position = -1
                signals[i] = -0.20
    
    return signals