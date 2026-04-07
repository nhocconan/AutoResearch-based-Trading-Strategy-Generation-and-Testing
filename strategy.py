#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily RSI with weekly trend filter and volume confirmation
# Uses RSI(14) for oversold/overbought conditions, weekly EMA(20) for trend direction,
# and volume spike (volume > 1.5x 20-day average) for confirmation.
# Designed for low trade frequency (target: 10-30 trades/year) to minimize fee drag.
# Works in bull markets via trend-following longs and in bear markets via mean-reversion shorts.

name = "daily_rsi_weekly_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate weekly EMA(20)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(30, n):  # Start after RSI and EMA warmup
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend direction from weekly EMA
        above_weekly_ema = close[i] > ema_1w_aligned[i]
        below_weekly_ema = close[i] < ema_1w_aligned[i]
        
        # RSI conditions
        oversold = rsi[i] < 30
        overbought = rsi[i] > 70
        
        # Long conditions: oversold + above weekly EMA + volume confirmation
        if oversold and above_weekly_ema and vol_confirmed:
            signals[i] = 0.25
        # Short conditions: overbought + below weekly EMA + volume confirmation
        elif overbought and below_weekly_ema and vol_confirmed:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals