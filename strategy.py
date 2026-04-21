#!/usr/bin/env python3
"""
1h strategy: 4h Trend + 1h Momentum + Volume Confirmation
- Use 4h EMA(21) for trend direction (long when price > EMA21, short when price < EMA21)
- Use 1h RSI(14) for momentum (long when RSI crosses above 50, short when crosses below 50)
- Use 1h volume > 1.5x 20-period average for confirmation
- Trade only during 08:00-20:00 UTC to avoid low-liquidity hours
- Fixed position size: 0.20 (20% of capital)
- Target: 15-30 trades/year to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # 4h EMA(21) for trend
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h RSI(14) for momentum
    delta = prices['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # 1h volume > 1.5x 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_4h_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        if position == 0:
            # Enter long: 4h uptrend + RSI crosses above 50 + volume confirmation
            if (price_close > ema_4h_aligned[i] and 
                rsi[i] > 50 and rsi[i-1] <= 50 and
                volume > 1.5 * vol_ma[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: 4h downtrend + RSI crosses below 50 + volume confirmation
            elif (price_close < ema_4h_aligned[i] and 
                  rsi[i] < 50 and rsi[i-1] >= 50 and
                  volume > 1.5 * vol_ma[i]):
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit: opposite RSI cross or trend change
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI crosses below 50 OR price < 4h EMA
                if rsi[i] < 50 and rsi[i-1] >= 50:
                    exit_signal = True
                elif price_close < ema_4h_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: RSI crosses above 50 OR price > 4h EMA
                if rsi[i] > 50 and rsi[i-1] <= 50:
                    exit_signal = True
                elif price_close > ema_4h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_4hEMA21_RSI14_Volume1.5x_Session"
timeframe = "1h"
leverage = 1.0