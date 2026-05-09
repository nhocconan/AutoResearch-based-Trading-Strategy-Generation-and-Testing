#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d trend and 12h momentum for entries.
# Uses 1d EMA34 for trend filter and 12h RSI(14) for momentum confirmation.
# Designed for low trade frequency (12-37/year) to avoid fee drag in 12h timeframe.
# Works in both bull/bear markets by requiring alignment with 1d trend and momentum confirmation.
name = "12h_EMA34_Trend_RSI14_Momentum"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h RSI(14) for momentum confirmation
    delta = pd.Series(close).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: spike above 1.5x 24-period average (2 days of 12h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24)  # Wait for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]  # Volume confirmation
        
        # Pre-compute hour for session filter (UTC 0-24)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Trade during active hours (8 AM - 8 PM UTC) to avoid low liquidity
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: price above 1d EMA34, RSI > 50 (bullish momentum), volume confirmation
            if (close[i] > ema_34_12h[i] and 
                rsi[i] > 50 and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: price below 1d EMA34, RSI < 50 (bearish momentum), volume confirmation
            elif (close[i] < ema_34_12h[i] and 
                  rsi[i] < 50 and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below 1d EMA34 or RSI < 40 (loss of momentum)
            if close[i] < ema_34_12h[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above 1d EMA34 or RSI > 60 (loss of momentum)
            if close[i] > ema_34_12h[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals