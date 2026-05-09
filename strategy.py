#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and volume confirmation.
# Uses 4h EMA(50) for trend direction and 1h RSI(14) for momentum entry.
# Volume filter requires volume > 1.5x 20-period EMA to confirm strength.
# Session filter (08-20 UTC) avoids low-liquidity periods.
# Designed to work in bull (pullbacks in uptrend) and bear (bounces in downtrend).
# Target: 15-35 trades/year to avoid fee drag.
name = "1h_EMA50_RSI14_Volume_Session"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h RSI(14) for momentum
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0 and session_filter[i]:
            # Enter long: price above 4h EMA50 + RSI < 30 (oversold) + volume confirmation
            if (price > ema_50_4h_aligned[i] and rsi[i] < 30 and vol_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: price below 4h EMA50 + RSI > 70 (overbought) + volume confirmation
            elif (price < ema_50_4h_aligned[i] and rsi[i] > 70 and vol_confirm[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 4h EMA50 or RSI > 70
            if price < ema_50_4h_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above 4h EMA50 or RSI < 30
            if price > ema_50_4h_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals